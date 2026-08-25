"""Stepped-Frequency Continuous Wave (SFCW) radar engine.

Orchestrates the bladeRF to sweep through discrete frequency steps,
capture IQ at each, and compute range profiles via IFFT.

Uses dual-channel reference: TX1+RX1 for antenna signal, TX2+RX2 as
phase reference (short cable loopback). Dividing signal by reference
eliminates random PLL phase offsets between TX and RX synthesizers.
"""

import json
import os
import threading
import time
import numpy as np

from bladerf_driver import BladeRFDriver
from bladerf._bladerf import libbladeRF, ffi
import bladerf

SPEED_OF_LIGHT = 299_792_458
CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'calibration')
GAIN_TABLE_PATH = os.path.join(CALIBRATION_DIR, 'gain_table.npz')


class SFCWEngine:
    def __init__(self, driver: BladeRFDriver):
        self.driver = driver
        self.start_freq = 1_000_000_000
        self.stop_freq = 2_380_000_000
        self.step_size = 10_000_000
        self.settle_time = 0.003
        self.num_buffers = 8
        self.range_offset = -0.13
        self.max_display_range = 3.0
        self.blank_range = 0.0
        self.coherent_avg = 4
        self.tx_headroom_db = 0
        # Gain table (loaded from disk)
        self._gain_table = None  # dict: freq_hz, tx_gain, rx_gain, tx2_scale, phase_std_deg
        self._load_gain_table()
        self.running = False
        self._stop_event = threading.Event()
        self._thread = None
        self._callback = None
        self._lock = threading.Lock()
        self._single_shot = False
        self._background = None
        self._capture_background = False
        self._reference = None
        self._capture_reference = False
        self._sub_mode = None  # 'background' | 'reference' | None
        self._cal_mode = None
        self._h_avg_accum = None
        self._h_avg_count = 0
        # Hardware calibration data (loaded from disk)
        self._cal_cable_thru = None
        self._cal_free_space = None
        self._cal_cable_thru_enabled = False
        self._load_hw_calibration()
        # Running mean subtraction state
        self._mean_accumulator = None
        self._mean_count = 0
        self._mean_subtraction_enabled = False

    # ------------------------------------------------------------------
    # Gain table management
    # ------------------------------------------------------------------

    def _load_gain_table(self):
        if not os.path.exists(GAIN_TABLE_PATH):
            self._gain_table = None
            return False
        try:
            npz = np.load(GAIN_TABLE_PATH)
            self._gain_table = {
                'freq_hz': npz['freq_hz'].astype(np.int64),
                'tx_gain': npz['tx_gain'].astype(int),
                'rx_gain': npz['rx_gain'].astype(int),
                'tx2_scale': npz['tx2_scale'].astype(np.float64),
                'phase_std_deg': npz['phase_std_deg'].astype(np.float64),
            }
            n = len(self._gain_table['freq_hz'])
            print(f"[sfcw] Loaded gain table ({n} entries, "
                  f"{self._gain_table['freq_hz'][0]/1e6:.0f}-{self._gain_table['freq_hz'][-1]/1e6:.0f} MHz)")
            return True
        except Exception as e:
            print(f"[sfcw] Failed to load gain table: {e}")
            self._gain_table = None
            return False

    def _save_gain_table(self, freq_hz, tx_gain, rx_gain, tx2_scale, phase_std_deg):
        os.makedirs(CALIBRATION_DIR, exist_ok=True)
        np.savez(GAIN_TABLE_PATH,
                 freq_hz=np.array(freq_hz, dtype=np.int64),
                 tx_gain=np.array(tx_gain, dtype=int),
                 rx_gain=np.array(rx_gain, dtype=int),
                 tx2_scale=np.array(tx2_scale, dtype=np.float64),
                 phase_std_deg=np.array(phase_std_deg, dtype=np.float64))
        self._gain_table = {
            'freq_hz': np.array(freq_hz, dtype=np.int64),
            'tx_gain': np.array(tx_gain, dtype=int),
            'rx_gain': np.array(rx_gain, dtype=int),
            'tx2_scale': np.array(tx2_scale, dtype=np.float64),
            'phase_std_deg': np.array(phase_std_deg, dtype=np.float64),
        }
        print(f"[sfcw] Saved gain table to {GAIN_TABLE_PATH}")

    def _lookup_table(self, freq_hz):
        """Find nearest table entry for a given frequency. Returns (tx_gain, rx_gain, tx2_scale)."""
        if self._gain_table is None:
            return None
        idx = np.argmin(np.abs(self._gain_table['freq_hz'] - freq_hz))
        return (
            int(self._gain_table['tx_gain'][idx]),
            int(self._gain_table['rx_gain'][idx]),
            float(self._gain_table['tx2_scale'][idx]),
        )

    # ------------------------------------------------------------------
    # Gain table generation
    # ------------------------------------------------------------------

    def generate_gain_table(self, callback=None):
        """Generate per-frequency gain lookup table. Runs in a thread.

        Tunes each frequency independently: ramps TX then RX to land RX1 at ~0.9,
        finds TX2 digital scale to land RX2 at ~0.9, measures phase stability.
        """
        if self.running:
            return False
        self._stop_event.clear()
        self.running = True
        self._thread = threading.Thread(
            target=self._generate_table_worker, args=(callback,), daemon=True)
        self._thread.start()
        return True

    def _generate_table_worker(self, callback):
        try:
            self._do_generate_table(callback)
        except Exception as e:
            print(f"[sfcw] Table generation error: {e}")
            if callback:
                callback({'type': 'error', 'message': str(e)})
        finally:
            self._stop_tx_rx()
            self.running = False

    def _do_generate_table(self, callback):
        start = 1_000_000_000
        stop = 6_000_000_000
        step = 10_000_000
        num_entries = int((stop - start) / step) + 1

        freq_hz_arr = np.linspace(start, stop, num_entries).astype(np.int64)
        tx_gain_arr = np.zeros(num_entries, dtype=int)
        rx_gain_arr = np.zeros(num_entries, dtype=int)
        tx2_scale_arr = np.zeros(num_entries, dtype=np.float64)
        phase_std_arr = np.zeros(num_entries, dtype=np.float64)

        # Configure and start TX/RX
        self.driver.set_waveform('cw', offset=100_000, amplitude=0.9)
        self.driver._configure_channels_dual()

        self._rx_latest = (None, None)
        self._rx_event = threading.Event()
        n = 1024
        t = np.arange(n, dtype=np.float64) / self.driver.sample_rate
        self._ref_tone = np.exp(-1j * 2 * np.pi * self.driver.cw_offset * t)

        self.driver.start_tx_dual(tx2_digital_scale=1.0)
        self.driver.start_rx_dual(self._rx_capture, num_samples=n)
        time.sleep(0.1)

        dev_ptr = self.driver.device.dev[0]
        tx_ch0 = bladerf.CHANNEL_TX(0)
        tx_ch1 = bladerf.CHANNEL_TX(1)
        rx_ch0 = bladerf.CHANNEL_RX(0)
        rx_ch1 = bladerf.CHANNEL_RX(1)

        libbladeRF.bladerf_set_gain_mode(dev_ptr, rx_ch0, libbladeRF.BLADERF_GAIN_MGC)
        libbladeRF.bladerf_set_gain_mode(dev_ptr, rx_ch1, libbladeRF.BLADERF_GAIN_MGC)

        # One-shot cal before gain table generation (tracking stays disabled)
        self.driver.run_oneshot_calibration()

        for i in range(num_entries):
            if self._stop_event.is_set():
                if callback:
                    callback({'type': 'error', 'message': 'Stopped by user'})
                return

            freq = int(freq_hz_arr[i])
            libbladeRF.bladerf_set_frequency(dev_ptr, tx_ch0, freq)
            libbladeRF.bladerf_set_frequency(dev_ptr, rx_ch0, freq)
            time.sleep(0.1)

            # Step 1: Ramp TX from 25 toward 66, measuring RX1
            tx_g = 25
            rx_g = 25
            libbladeRF.bladerf_set_gain(dev_ptr, rx_ch0, rx_g)
            libbladeRF.bladerf_set_gain(dev_ptr, rx_ch1, rx_g)

            best_tx = tx_g
            best_rx = rx_g
            rx1_mag = 0.0

            # Ramp TX
            while tx_g <= 66:
                libbladeRF.bladerf_set_gain(dev_ptr, tx_ch0, tx_g)
                libbladeRF.bladerf_set_gain(dev_ptr, tx_ch1, tx_g)
                time.sleep(0.1)
                _, _, rx1_peak, _ = self._measure_step(4)
                rx1_mag = rx1_peak
                if rx1_mag >= 0.9:
                    best_tx = tx_g
                    break
                tx_g += 1
            else:
                best_tx = 66

            # Step 2: If TX maxed and RX1 < 0.9, ramp RX
            if rx1_mag < 0.9 and best_tx >= 66:
                rx_g = 26
                while rx_g <= 60:
                    libbladeRF.bladerf_set_gain(dev_ptr, rx_ch0, rx_g)
                    libbladeRF.bladerf_set_gain(dev_ptr, rx_ch1, rx_g)
                    time.sleep(0.1)
                    _, _, rx1_peak, _ = self._measure_step(4)
                    rx1_mag = rx1_peak
                    if rx1_mag >= 0.9:
                        best_rx = rx_g
                        break
                    rx_g += 1
                else:
                    best_rx = 60
            else:
                best_rx = rx_g

            # Step 3: Back off TX if overshooting
            if rx1_mag > 0.95 and best_tx > 25:
                while best_tx > 25 and rx1_mag > 0.95:
                    best_tx -= 1
                    libbladeRF.bladerf_set_gain(dev_ptr, tx_ch0, best_tx)
                    libbladeRF.bladerf_set_gain(dev_ptr, tx_ch1, best_tx)
                    time.sleep(0.1)
                    _, _, rx1_peak, _ = self._measure_step(4)
                    rx1_mag = rx1_peak

            # Ensure final gains are applied
            libbladeRF.bladerf_set_gain(dev_ptr, tx_ch0, best_tx)
            libbladeRF.bladerf_set_gain(dev_ptr, tx_ch1, best_tx)
            libbladeRF.bladerf_set_gain(dev_ptr, rx_ch0, best_rx)
            libbladeRF.bladerf_set_gain(dev_ptr, rx_ch1, best_rx)
            time.sleep(0.1)

            # Step 4: Find TX2 digital scale to land RX2 at ~0.9
            # Binary search: scale down from 1.0
            scale_lo = 0.001
            scale_hi = 1.0
            best_scale = 0.05  # fallback

            for _ in range(12):
                mid = (scale_lo + scale_hi) / 2.0
                self.driver._tx2_digital_scale = mid
                time.sleep(0.1)
                _, _, _, rx2_peak = self._measure_step(4)
                if rx2_peak > 0.9:
                    scale_hi = mid
                else:
                    scale_lo = mid
                if abs(rx2_peak - 0.9) < 0.03:
                    best_scale = mid
                    break
            else:
                best_scale = (scale_lo + scale_hi) / 2.0

            self.driver._tx2_digital_scale = best_scale
            time.sleep(0.05)

            # Step 5: Measure phase stability (20 captures)
            phases = []
            for _ in range(20):
                self._rx_event.clear()
                self._rx_event.wait(timeout=1.0)
                sig, ref, _, _ = self._measure_step(2)
                if abs(ref) > 1e-10:
                    phases.append(np.angle(sig / ref))
            if len(phases) >= 2:
                phase_std = float(np.degrees(np.std(phases)))
            else:
                phase_std = 999.0

            # Record
            tx_gain_arr[i] = best_tx
            rx_gain_arr[i] = best_rx
            tx2_scale_arr[i] = best_scale
            phase_std_arr[i] = phase_std

            # Final magnitude check
            _, _, rx1_final, rx2_final = self._measure_step(4)

            print(f"[cal] {i+1}/{num_entries} — {freq/1e6:.0f} MHz: "
                  f"TX={best_tx}, RX={best_rx}, scale={best_scale:.4f}, "
                  f"mag={rx1_final:.3f}/{rx2_final:.3f}, phase_std={phase_std:.1f}°")

            if callback:
                callback({
                    'type': 'progress',
                    'step': i,
                    'total': num_entries,
                    'freq_mhz': freq / 1e6,
                    'tx_gain': best_tx,
                    'rx_gain': best_rx,
                    'rx1_mag': float(rx1_final),
                    'phase_std_deg': phase_std,
                })

        # Save table
        self._save_gain_table(freq_hz_arr, tx_gain_arr, rx_gain_arr, tx2_scale_arr, phase_std_arr)

        if callback:
            callback({
                'type': 'table_complete',
                'num_entries': num_entries,
                'freq_range_mhz': [start / 1e6, stop / 1e6],
                'tx_range': [int(tx_gain_arr.min()), int(tx_gain_arr.max())],
                'rx_range': [int(rx_gain_arr.min()), int(rx_gain_arr.max())],
                'phase_std_median': float(np.median(phase_std_arr)),
            })

    # ------------------------------------------------------------------
    # Gain table verification
    # ------------------------------------------------------------------

    def verify_gain_table(self, callback=None):
        """Verify gain table by sweeping all entries and measuring actual levels."""
        if self.running:
            return False
        if self._gain_table is None:
            if callback:
                callback({'type': 'error', 'message': 'No gain table loaded'})
            return False
        self._stop_event.clear()
        self.running = True
        self._thread = threading.Thread(
            target=self._verify_table_worker, args=(callback,), daemon=True)
        self._thread.start()
        return True

    def _verify_table_worker(self, callback):
        try:
            self._do_verify_table(callback)
        except Exception as e:
            print(f"[sfcw] Table verification error: {e}")
            if callback:
                callback({'type': 'error', 'message': str(e)})
        finally:
            self._stop_tx_rx()
            self.running = False

    def _do_verify_table(self, callback):
        tbl = self._gain_table
        num_entries = len(tbl['freq_hz'])

        self.driver.set_waveform('cw', offset=100_000, amplitude=0.9)
        self.driver._configure_channels_dual()

        self._rx_latest = (None, None)
        self._rx_event = threading.Event()
        n = 1024
        t = np.arange(n, dtype=np.float64) / self.driver.sample_rate
        self._ref_tone = np.exp(-1j * 2 * np.pi * self.driver.cw_offset * t)

        self.driver.start_tx_dual(tx2_digital_scale=1.0)
        self.driver.start_rx_dual(self._rx_capture, num_samples=n)
        time.sleep(0.1)

        dev_ptr = self.driver.device.dev[0]
        tx_ch0 = bladerf.CHANNEL_TX(0)
        tx_ch1 = bladerf.CHANNEL_TX(1)
        rx_ch0 = bladerf.CHANNEL_RX(0)
        rx_ch1 = bladerf.CHANNEL_RX(1)

        libbladeRF.bladerf_set_gain_mode(dev_ptr, rx_ch0, libbladeRF.BLADERF_GAIN_MGC)
        libbladeRF.bladerf_set_gain_mode(dev_ptr, rx_ch1, libbladeRF.BLADERF_GAIN_MGC)

        self.driver.run_oneshot_calibration()

        rx1_mags = np.zeros(num_entries)
        rx2_mags = np.zeros(num_entries)
        phase_stds = np.zeros(num_entries)
        clipped = []
        problems = []

        for i in range(num_entries):
            if self._stop_event.is_set():
                return

            freq = int(tbl['freq_hz'][i])
            tx_g = int(tbl['tx_gain'][i])
            rx_g = int(tbl['rx_gain'][i])
            scale = float(tbl['tx2_scale'][i])

            libbladeRF.bladerf_set_frequency(dev_ptr, tx_ch0, freq)
            libbladeRF.bladerf_set_frequency(dev_ptr, rx_ch0, freq)
            libbladeRF.bladerf_set_gain(dev_ptr, tx_ch0, tx_g)
            libbladeRF.bladerf_set_gain(dev_ptr, tx_ch1, tx_g)
            libbladeRF.bladerf_set_gain(dev_ptr, rx_ch0, rx_g)
            libbladeRF.bladerf_set_gain(dev_ptr, rx_ch1, rx_g)
            self.driver._tx2_digital_scale = scale
            time.sleep(0.1)

            _, _, rx1_peak, rx2_peak = self._measure_step(4)
            rx1_mags[i] = rx1_peak
            rx2_mags[i] = rx2_peak

            if rx1_peak > 0.98 or rx2_peak > 0.98:
                clipped.append(i)

            # Phase stability
            phases = []
            for _ in range(10):
                self._rx_event.clear()
                self._rx_event.wait(timeout=1.0)
                sig, ref, _, _ = self._measure_step(2)
                if abs(ref) > 1e-10:
                    phases.append(np.angle(sig / ref))
            phase_stds[i] = float(np.degrees(np.std(phases))) if len(phases) >= 2 else 999.0

            if rx1_peak < 0.6 or rx1_peak > 0.98 or phase_stds[i] > 10.0:
                problems.append(i)

            if callback and i % 10 == 0:
                callback({
                    'type': 'progress',
                    'step': i,
                    'total': num_entries,
                    'freq_mhz': freq / 1e6,
                })

        # Summary
        rx1_in_range = np.sum((rx1_mags >= 0.8) & (rx1_mags <= 0.95))
        rx2_in_range = np.sum((rx2_mags >= 0.8) & (rx2_mags <= 0.95))
        high_phase = np.sum(phase_stds > 10.0)

        summary = {
            'type': 'verify_complete',
            'num_entries': num_entries,
            'rx1_in_range': int(rx1_in_range),
            'rx2_in_range': int(rx2_in_range),
            'rx1_range': [float(rx1_mags.min()), float(rx1_mags.max())],
            'rx2_range': [float(rx2_mags.min()), float(rx2_mags.max())],
            'high_phase_count': int(high_phase),
            'clipped_count': len(clipped),
            'problem_count': len(problems),
            'phase_std_median': float(np.median(phase_stds)),
        }

        print(f"[sfcw] Verify: {num_entries} entries, "
              f"RX1 in [0.8,0.95]: {rx1_in_range}/{num_entries}, "
              f"RX2 in [0.8,0.95]: {rx2_in_range}/{num_entries}, "
              f"phase>10°: {high_phase}, clipped: {len(clipped)}")

        if problems:
            print(f"[sfcw] Problem frequencies ({len(problems)}):")
            for idx in problems[:20]:
                print(f"  {tbl['freq_hz'][idx]/1e6:.0f} MHz: "
                      f"RX1={rx1_mags[idx]:.3f}, phase={phase_stds[idx]:.1f}°")

        if callback:
            callback(summary)

    # ------------------------------------------------------------------
    # Hardware calibration loading
    # ------------------------------------------------------------------

    def _load_hw_calibration(self):
        for mode, attr in [('cable_thru', '_cal_cable_thru'), ('free_space', '_cal_free_space')]:
            filepath = os.path.join(CALIBRATION_DIR, f'{mode}.npz')
            if os.path.exists(filepath):
                try:
                    npz = np.load(filepath, allow_pickle=True)
                    setattr(self, attr, {
                        'frequencies': npz['frequencies'],
                        'h_complex': npz['h_complex'],
                    })
                    print(f"[sfcw] Loaded {mode} calibration ({len(npz['frequencies'])} points)")
                except Exception as e:
                    print(f"[sfcw] Failed to load {mode} calibration: {e}")
                    setattr(self, attr, None)

    def _interpolate_cal(self, cal_data, target_freqs):
        cal_freqs = cal_data['frequencies']
        cal_h = cal_data['h_complex']
        cal_mag = np.abs(cal_h)
        cal_phase = np.unwrap(np.angle(cal_h))
        mag_interp = np.interp(target_freqs, cal_freqs, cal_mag)
        phase_interp = np.interp(target_freqs, cal_freqs, cal_phase)
        return mag_interp * np.exp(1j * phase_interp)


    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_steps(self):
        return int((self.stop_freq - self.start_freq) / self.step_size) + 1

    @property
    def bandwidth(self):
        return self.stop_freq - self.start_freq

    @property
    def range_resolution(self):
        if self.bandwidth == 0:
            return float('inf')
        return SPEED_OF_LIGHT / (2 * self.bandwidth)

    @property
    def max_range(self):
        if self.step_size == 0:
            return float('inf')
        return SPEED_OF_LIGHT / (2 * self.step_size)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def set_params(self, **kwargs):
        with self._lock:
            if 'start_freq' in kwargs:
                self.start_freq = int(kwargs['start_freq'])
            if 'stop_freq' in kwargs:
                self.stop_freq = int(kwargs['stop_freq'])
            if 'step_size' in kwargs:
                self.step_size = int(kwargs['step_size'])
            if 'settle_time' in kwargs:
                self.settle_time = float(kwargs['settle_time'])
            if 'num_buffers' in kwargs:
                self.num_buffers = max(1, int(kwargs['num_buffers']))
            if 'range_offset' in kwargs:
                self.range_offset = float(kwargs['range_offset'])
            if 'max_display_range' in kwargs:
                self.max_display_range = float(kwargs['max_display_range'])
            if 'blank_range' in kwargs:
                self.blank_range = float(kwargs['blank_range'])
            if 'coherent_avg' in kwargs:
                self.coherent_avg = max(1, int(kwargs['coherent_avg']))
                self._h_avg_accum = None
                self._h_avg_count = 0
            if 'tx_headroom_db' in kwargs:
                self.tx_headroom_db = max(0, int(kwargs['tx_headroom_db']))

    def get_params(self):
        return {
            'start_freq': self.start_freq,
            'stop_freq': self.stop_freq,
            'step_size': self.step_size,
            'settle_time': self.settle_time,
            'num_buffers': self.num_buffers,
            'range_offset': self.range_offset,
            'max_display_range': self.max_display_range,
            'blank_range': self.blank_range,
            'coherent_avg': self.coherent_avg,
            'num_steps': self.num_steps,
            'bandwidth': self.bandwidth,
            'range_resolution': self.range_resolution,
            'max_range': self.max_range,
            'background_active': self._background is not None,
            'reference_active': self._reference is not None,
            'sub_mode': self._sub_mode,
            'mean_subtraction': self._mean_subtraction_enabled,
            'mean_count': self._mean_count,
            'gain_table_loaded': self._gain_table is not None,
            'gain_table_entries': len(self._gain_table['freq_hz']) if self._gain_table else 0,
            'tx_headroom_db': self.tx_headroom_db,
        }

    # ------------------------------------------------------------------
    # Subtraction controls
    # ------------------------------------------------------------------

    def capture_background(self):
        self._capture_background = True

    def clear_background(self):
        self._background = None
        self._capture_background = False
        self._sub_mode = None

    def capture_reference(self):
        self._capture_reference = True

    def clear_reference(self):
        self._reference = None
        self._capture_reference = False
        self._sub_mode = None

    def clear_all_subtraction(self):
        self._background = None
        self._capture_background = False
        self._reference = None
        self._capture_reference = False
        self._sub_mode = None

    def enable_mean_subtraction(self):
        self._mean_subtraction_enabled = True
        self._mean_accumulator = None
        self._mean_count = 0

    def disable_mean_subtraction(self):
        self._mean_subtraction_enabled = False

    def reset_mean(self):
        self._mean_accumulator = None
        self._mean_count = 0

    # ------------------------------------------------------------------
    # Sweep control
    # ------------------------------------------------------------------

    def start(self, callback):
        if self.running:
            return
        self._callback = callback
        self._stop_event.clear()
        self._single_shot = False
        self.running = True
        self._thread = threading.Thread(target=self._sweep_loop, daemon=True)
        self._thread.start()

    def run_single(self, callback):
        if self.running:
            self.stop()
        self._callback = callback
        self._stop_event.clear()
        self._single_shot = True
        self.running = True
        self._thread = threading.Thread(target=self._sweep_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.running:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        self.running = False

    def _sweep_loop(self):
        try:
            self._configure_hardware()
            self._start_tx_rx()

            while not self._stop_event.is_set():
                range_profile = self._perform_sweep()
                if range_profile is not None and self._callback:
                    self._callback(range_profile)
                if self._single_shot:
                    break

        except Exception as e:
            print(f"[sfcw] Sweep error: {e}")
            if self._callback:
                self._callback({'error': str(e)})
        finally:
            self._stop_tx_rx()
            self.running = False

    def _configure_hardware(self):
        self.driver.set_waveform('cw', offset=100_000, amplitude=0.9)
        self.driver._configure_channels_dual()

    def _start_tx_rx(self):
        self._rx_latest = (None, None)
        self._rx_event = threading.Event()
        n = 1024
        t = np.arange(n, dtype=np.float64) / self.driver.sample_rate
        self._ref_tone = np.exp(-1j * 2 * np.pi * self.driver.cw_offset * t)
        # Start with default scale; will be updated per-step from table
        self.driver.start_tx_dual(tx2_digital_scale=0.05)
        self.driver.start_rx_dual(self._rx_capture, num_samples=n)
        time.sleep(0.05)

        dev_ptr = self.driver.device.dev[0]
        libbladeRF.bladerf_set_gain_mode(dev_ptr, bladerf.CHANNEL_RX(0), libbladeRF.BLADERF_GAIN_MGC)
        libbladeRF.bladerf_set_gain_mode(dev_ptr, bladerf.CHANNEL_RX(1), libbladeRF.BLADERF_GAIN_MGC)

        # One-shot cal at sweep center freq before starting (tracking stays disabled)
        self.driver.run_oneshot_calibration()

    def _stop_tx_rx(self):
        self.driver.stop_rx_dual()
        self.driver.stop_tx_dual()

    def _rx_capture(self, rx1_iq, rx2_iq):
        self._rx_latest = (rx1_iq, rx2_iq)
        self._rx_event.set()

    def _measure_step(self, num_buffers):
        """Capture IQ at current frequency, return (sig_complex, ref_complex, rx1_peak, rx2_peak)."""
        sig_accum = 0j
        ref_accum = 0j
        rx1_peak = 0.0
        rx2_peak = 0.0
        captured = 0
        for _ in range(num_buffers):
            self._rx_event.clear()
            if not self._rx_event.wait(timeout=1.0):
                break

            rx1, rx2 = self._rx_latest
            if rx1 is None or rx2 is None:
                continue

            i1 = rx1[0::2].astype(np.float64) / 2047.0
            q1 = rx1[1::2].astype(np.float64) / 2047.0
            sig_accum += np.mean((i1 + 1j * q1) * self._ref_tone)
            rx1_peak = max(rx1_peak, float(np.max(np.abs(i1))), float(np.max(np.abs(q1))))

            i2 = rx2[0::2].astype(np.float64) / 2047.0
            q2 = rx2[1::2].astype(np.float64) / 2047.0
            ref_accum += np.mean((i2 + 1j * q2) * self._ref_tone)
            rx2_peak = max(rx2_peak, float(np.max(np.abs(i2))), float(np.max(np.abs(q2))))

            captured += 1

        sig = sig_accum / max(captured, 1)
        ref = ref_accum / max(captured, 1)
        return sig, ref, rx1_peak, rx2_peak

    # ------------------------------------------------------------------
    # Sweep execution
    # ------------------------------------------------------------------

    def _good_freq_mask(self, freqs):
        """Return boolean mask of frequencies with stable phase during calibration."""
        if self._gain_table is None:
            return np.ones(len(freqs), dtype=bool)
        tbl_freqs = self._gain_table['freq_hz']
        tbl_phase = self._gain_table['phase_std_deg']
        mask = np.ones(len(freqs), dtype=bool)
        for i, f in enumerate(freqs):
            idx = np.argmin(np.abs(tbl_freqs - f))
            if tbl_phase[idx] > 5.0:
                mask[i] = False
        return mask

    def _ndft(self, h_cal, freqs, num_range_bins):
        """Non-uniform DFT: compute range profile from arbitrary frequency samples.

        Matched filter: for each candidate range bin, correlate H(f) with the
        expected phase progression exp(+j*2pi*f*2d/c). This is exact regardless
        of frequency spacing.
        """
        max_range = SPEED_OF_LIGHT / (2 * self.step_size)
        ranges = np.linspace(0, max_range, num_range_bins)
        tau = 2 * ranges / SPEED_OF_LIGHT

        kernel = np.exp(+1j * 2 * np.pi * freqs[None, :] * tau[:, None])
        window = np.hanning(len(freqs))
        range_profile = kernel @ (h_cal * window) / len(freqs)
        return range_profile, ranges

    def _perform_sweep(self):
        with self._lock:
            start = self.start_freq
            stop = self.stop_freq
            step = self.step_size
            settle = self.settle_time
            num_buffers = self.num_buffers
            max_disp = self.max_display_range
            avg_count = self.coherent_avg

        num_steps = int((stop - start) / step) + 1
        freqs = np.linspace(start, stop, num_steps).astype(np.int64)
        h_signal = np.zeros(num_steps, dtype=np.complex128)
        h_reference = np.zeros(num_steps, dtype=np.complex128)
        clipped = np.zeros(num_steps, dtype=bool)

        dev_ptr = self.driver.device.dev[0]
        tx_ch = bladerf.CHANNEL_TX(0)
        tx_ch1 = bladerf.CHANNEL_TX(1)
        rx_ch = bladerf.CHANNEL_RX(0)
        rx_ch1 = bladerf.CHANNEL_RX(1)

        has_table = self._gain_table is not None

        for i in range(num_steps):
            if self._stop_event.is_set():
                return None

            f = int(freqs[i])

            libbladeRF.bladerf_set_frequency(dev_ptr, tx_ch, f)
            libbladeRF.bladerf_set_frequency(dev_ptr, rx_ch, f)

            if has_table:
                tx_g, rx_g, scale = self._lookup_table(f)
                libbladeRF.bladerf_set_gain(dev_ptr, tx_ch, tx_g)
                libbladeRF.bladerf_set_gain(dev_ptr, tx_ch1, tx_g)
                libbladeRF.bladerf_set_gain(dev_ptr, rx_ch, rx_g)
                libbladeRF.bladerf_set_gain(dev_ptr, rx_ch1, rx_g)
                self.driver._tx2_digital_scale = scale

            time.sleep(settle)

            # Discard first buffer after freq/gain change (may contain transient)
            self._rx_event.clear()
            self._rx_event.wait(timeout=1.0)

            sig, ref, rx1_peak, rx2_peak = self._measure_step(num_buffers)

            # Validate: no clipping, and phase is stable (two measurements agree)
            valid_bin = True
            if rx1_peak > 0.98 or rx2_peak > 0.98:
                valid_bin = False
            elif abs(ref) > 1e-10:
                # Quick phase check: take a second measurement
                sig2, ref2, _, _ = self._measure_step(num_buffers)
                if abs(ref2) > 1e-10:
                    h1 = sig / ref
                    h2 = sig2 / ref2
                    phase_diff = abs(np.angle(h2 / h1))
                    if phase_diff > 0.087:  # ~5 degrees
                        valid_bin = False

            # Retry up to 2 times if invalid
            if not valid_bin:
                for _retry in range(2):
                    if has_table and rx2_peak > 0.98:
                        self.driver._tx2_digital_scale = self.driver._tx2_digital_scale * 0.5
                    time.sleep(settle)
                    self._rx_event.clear()
                    self._rx_event.wait(timeout=1.0)
                    sig, ref, rx1_peak, rx2_peak = self._measure_step(num_buffers)
                    if rx1_peak > 0.98 or rx2_peak > 0.98:
                        continue
                    if abs(ref) > 1e-10:
                        sig2, ref2, _, _ = self._measure_step(num_buffers)
                        if abs(ref2) > 1e-10:
                            h1 = sig / ref
                            h2 = sig2 / ref2
                            phase_diff = abs(np.angle(h2 / h1))
                            if phase_diff <= 0.087:
                                valid_bin = True
                                break

            if not valid_bin:
                clipped[i] = True

            h_signal[i] = sig
            h_reference[i] = ref

            if self._callback and i % 10 == 0:
                self._callback({
                    'type': 'progress',
                    'step': i,
                    'total': num_steps,
                    'freq_mhz': freqs[i] / 1e6,
                })

        # Phase-reference division: cancels PLL phase noise + TX/RX gain.
        # Result: h_cal = antenna_H(f) / (tx2_scale(f) * cable_H(f))
        ref_mag = np.abs(h_reference)
        valid = ref_mag > 1e-10
        h_cal = np.zeros(num_steps, dtype=np.complex128)
        h_cal[valid] = h_signal[valid] / h_reference[valid]

        # Coherent averaging on raw division (before normalization).
        # Averaging complex phasors improves SNR: signal bins stay coherent,
        # noise bins cancel. Do NOT normalize before averaging.
        if self._h_avg_accum is None or len(self._h_avg_accum) != num_steps:
            self._h_avg_accum = h_cal.copy()
            self._h_avg_count = 1
        else:
            self._h_avg_count += 1
            if self._h_avg_count > avg_count:
                alpha = 1.0 / avg_count
                self._h_avg_accum = (1 - alpha) * self._h_avg_accum + alpha * h_cal
            else:
                self._h_avg_accum = (self._h_avg_accum * (self._h_avg_count - 1) + h_cal) / self._h_avg_count

        h_averaged = self._h_avg_accum.copy()

        # Capture reference (wall-aligned subtraction)
        if self._capture_reference:
            self._reference = h_averaged.copy()
            self._capture_reference = False
            self._background = None
            self._sub_mode = 'reference'

        # Capture background (static subtraction)
        if self._capture_background:
            self._background = h_averaged.copy()
            self._capture_background = False
            self._reference = None
            self._sub_mode = 'background'

        # Apply subtraction on full grid (before filtering)
        h_proc = h_averaged
        if self._sub_mode == 'background' and self._background is not None and len(self._background) == num_steps:
            h_proc = h_averaged - self._background
        elif self._sub_mode == 'reference' and self._reference is not None and len(self._reference) == num_steps:
            h_proc = h_averaged - self._reference

        # Scale compensation removed: with wall-present calibration and no headroom,
        # the sig/ref division directly gives the scene transfer function.
        # The per-frequency scale only prevents RX2 clipping — it divides out in sig/ref.

        # Filter: combine calibration quality mask with runtime clipping/phase check
        good_mask = self._good_freq_mask(freqs) & ~clipped
        good_freqs = freqs[good_mask]
        h_good = h_proc[good_mask]
        num_good = int(np.sum(good_mask))

        # Phase coherence diagnostics (on good frequencies only)
        phase_unwrapped = np.unwrap(np.angle(h_good))
        coeffs = np.polyfit(np.arange(num_good), phase_unwrapped, 1)
        residuals = phase_unwrapped - np.polyval(coeffs, np.arange(num_good))
        phase_std = float(np.std(residuals))

        # Range profile via NDFT (uses actual frequencies for correct phase modeling)
        num_range_bins = 512
        range_profile, distances = self._ndft(h_good, good_freqs, num_range_bins)
        distances = distances - self.range_offset

        magnitude_linear = np.abs(range_profile)
        magnitude_db = 20 * np.log10(magnitude_linear + 1e-12)

        half = num_range_bins // 2
        magnitude_db = magnitude_db[:half]
        magnitude_linear = magnitude_linear[:half]
        distances = distances[:half]

        # Clip to blank_range..max_display_range
        blank = self.blank_range
        display_mask = (distances >= blank) & (distances <= max_disp)
        magnitude_db = magnitude_db[display_mask]
        magnitude_linear = magnitude_linear[display_mask]
        distances = distances[display_mask]

        # Peak detection
        if len(magnitude_db) > 0:
            peak_idx = int(np.argmax(magnitude_db))
            peak_db = float(magnitude_db[peak_idx])
            peak_dist = float(distances[peak_idx])
            noise_floor = float(np.median(np.sort(magnitude_db)[:len(magnitude_db) // 2]))
            snr = peak_db - noise_floor
        else:
            peak_idx = 0
            peak_db = -100.0
            peak_dist = 0.0
            noise_floor = -100.0
            snr = 0.0

        result = {
            'type': 'range_profile',
            'distances': distances.tolist(),
            'magnitudes': magnitude_db.tolist(),
            'magnitudes_linear': magnitude_linear.tolist(),
            'range_resolution': SPEED_OF_LIGHT / (2 * (stop - start)),
            'max_range': max_disp,
            'num_steps': num_steps,
            'num_good': num_good,
            'avg_count': min(self._h_avg_count, avg_count),
            'timestamp': time.time(),
            'peak': {
                'distance_m': peak_dist,
                'magnitude_db': peak_db,
                'snr_db': snr,
                'noise_floor_db': noise_floor,
            },
            'phase_coherence': {
                'phase_std_rad': phase_std,
                'phase_std_deg': float(np.degrees(phase_std)),
                'coherent': phase_std < 0.3,
                'slope_rad_per_step': float(coeffs[0]),
            },
        }

        return result

    # ------------------------------------------------------------------
    # Hardware calibration
    # ------------------------------------------------------------------

    def run_calibration(self, mode, callback):
        """Run a calibration sweep. mode: 'cable_thru', 'free_space', 'per_position'."""
        if self.running:
            self.stop()
        self._callback = callback
        self._stop_event.clear()
        self._single_shot = True
        self._cal_mode = mode
        self.running = True
        self._thread = threading.Thread(target=self._calibration_loop, daemon=True)
        self._thread.start()

    def _calibration_loop(self):
        try:
            self._configure_hardware_for_cal()
            self._start_tx_rx_for_cal()

            result = self._perform_calibration_sweep()
            if result is not None and self._callback:
                self._callback(result)

        except Exception as e:
            print(f"[sfcw] Calibration error: {e}")
            if self._callback:
                self._callback({'error': str(e)})
        finally:
            self._stop_tx_rx()
            self.running = False
            self._cal_mode = None

    def _configure_hardware_for_cal(self):
        self.driver.set_waveform('cw', offset=100_000, amplitude=0.9)
        self.driver._configure_channels_dual()

    def _start_tx_rx_for_cal(self):
        self._rx_latest = (None, None)
        self._rx_event = threading.Event()
        n = 1024
        t = np.arange(n, dtype=np.float64) / self.driver.sample_rate
        self._ref_tone = np.exp(-1j * 2 * np.pi * self.driver.cw_offset * t)
        self.driver.start_tx_dual(tx2_digital_scale=0.05)
        self.driver.start_rx_dual(self._rx_capture, num_samples=n)
        time.sleep(0.05)

        dev_ptr = self.driver.device.dev[0]
        libbladeRF.bladerf_set_gain_mode(dev_ptr, bladerf.CHANNEL_RX(0), libbladeRF.BLADERF_GAIN_MGC)
        libbladeRF.bladerf_set_gain_mode(dev_ptr, bladerf.CHANNEL_RX(1), libbladeRF.BLADERF_GAIN_MGC)

        self.driver.run_oneshot_calibration()

    def _perform_calibration_sweep(self):
        """Perform a single sweep collecting raw complex data for calibration."""
        mode = self._cal_mode

        with self._lock:
            start = self.start_freq
            stop = self.stop_freq
            step = self.step_size
            settle = self.settle_time
            num_buffers = self.num_buffers

        num_steps = int((stop - start) / step) + 1
        freqs = np.linspace(start, stop, num_steps).astype(np.int64)
        h_signal = np.zeros(num_steps, dtype=np.complex128)
        h_reference = np.zeros(num_steps, dtype=np.complex128)

        dev_ptr = self.driver.device.dev[0]
        tx_ch = bladerf.CHANNEL_TX(0)
        tx_ch1 = bladerf.CHANNEL_TX(1)
        rx_ch = bladerf.CHANNEL_RX(0)
        rx_ch1 = bladerf.CHANNEL_RX(1)

        has_table = self._gain_table is not None

        for i in range(num_steps):
            if self._stop_event.is_set():
                return None

            f = int(freqs[i])

            libbladeRF.bladerf_set_frequency(dev_ptr, tx_ch, f)
            libbladeRF.bladerf_set_frequency(dev_ptr, rx_ch, f)

            if has_table:
                tx_g, rx_g, scale = self._lookup_table(f)
                libbladeRF.bladerf_set_gain(dev_ptr, tx_ch, tx_g)
                libbladeRF.bladerf_set_gain(dev_ptr, tx_ch1, tx_g)
                libbladeRF.bladerf_set_gain(dev_ptr, rx_ch, rx_g)
                libbladeRF.bladerf_set_gain(dev_ptr, rx_ch1, rx_g)
                self.driver._tx2_digital_scale = scale

            time.sleep(settle)

            # Discard first buffer after freq/gain change
            self._rx_event.clear()
            self._rx_event.wait(timeout=1.0)

            sig_accum = 0j
            ref_accum = 0j
            captured = 0
            for _ in range(num_buffers):
                self._rx_event.clear()
                if not self._rx_event.wait(timeout=1.0):
                    break

                rx1, rx2 = self._rx_latest
                if rx1 is None or rx2 is None:
                    continue

                i1 = rx1[0::2].astype(np.float64) / 2047.0
                q1 = rx1[1::2].astype(np.float64) / 2047.0
                sig_accum += np.mean((i1 + 1j * q1) * self._ref_tone)

                i2 = rx2[0::2].astype(np.float64) / 2047.0
                q2 = rx2[1::2].astype(np.float64) / 2047.0
                ref_accum += np.mean((i2 + 1j * q2) * self._ref_tone)

                captured += 1

            h_signal[i] = sig_accum / max(captured, 1)
            h_reference[i] = ref_accum / max(captured, 1)

            if self._callback and i % 10 == 0:
                self._callback({
                    'type': 'progress',
                    'step': i,
                    'total': num_steps,
                    'freq_mhz': freqs[i] / 1e6,
                })

        # Phase-reference division
        ref_mag = np.abs(h_reference)
        valid = ref_mag > 1e-10
        h_cal = np.zeros(num_steps, dtype=np.complex128)
        h_cal[valid] = h_signal[valid] / h_reference[valid]

        timestamp = time.time()

        return {
            'type': 'calibration_raw',
            'mode': mode,
            'frequencies': freqs,
            'h_complex': h_cal,
            'h_signal_raw': h_signal,
            'h_reference_raw': h_reference,
            'timestamp': timestamp,
        }

    def save_calibration(self, mode, data, step_size_cm=None):
        """Save calibration data to .npz file."""
        os.makedirs(CALIBRATION_DIR, exist_ok=True)

        params = {
            'start_freq': self.start_freq,
            'stop_freq': self.stop_freq,
            'step_size': self.step_size,
            'settle_time': self.settle_time,
            'num_buffers': self.num_buffers,
            'mode': mode,
        }

        filepath = os.path.join(CALIBRATION_DIR, f'{mode}.npz')

        if mode == 'per_position':
            existing = self._load_per_position_data()
            if existing is not None:
                h_complex_list = np.vstack([existing['h_complex'], data['h_complex'][np.newaxis, :]])
                h_signal_list = np.vstack([existing['h_signal_raw'], data['h_signal_raw'][np.newaxis, :]])
                h_reference_list = np.vstack([existing['h_reference_raw'], data['h_reference_raw'][np.newaxis, :]])
                timestamps = np.append(existing['timestamp'], data['timestamp'])
            else:
                h_complex_list = data['h_complex'][np.newaxis, :]
                h_signal_list = data['h_signal_raw'][np.newaxis, :]
                h_reference_list = data['h_reference_raw'][np.newaxis, :]
                timestamps = np.array([data['timestamp']])

            save_kwargs = {
                'frequencies': data['frequencies'],
                'h_complex': h_complex_list,
                'h_signal_raw': h_signal_list,
                'h_reference_raw': h_reference_list,
                'params': json.dumps(params),
                'timestamp': timestamps,
            }
            if step_size_cm is not None:
                save_kwargs['step_size_cm'] = np.array(step_size_cm)
            np.savez(filepath, **save_kwargs)
        else:
            np.savez(filepath,
                     frequencies=data['frequencies'],
                     h_complex=data['h_complex'],
                     h_signal_raw=data['h_signal_raw'],
                     h_reference_raw=data['h_reference_raw'],
                     params=json.dumps(params),
                     timestamp=np.array(data['timestamp']))

        print(f"[sfcw] Calibration saved: {filepath}")
        self._load_hw_calibration()

    def _load_per_position_data(self):
        filepath = os.path.join(CALIBRATION_DIR, 'per_position.npz')
        if not os.path.exists(filepath):
            return None
        try:
            npz = np.load(filepath, allow_pickle=True)
            return {
                'h_complex': npz['h_complex'],
                'h_signal_raw': npz['h_signal_raw'],
                'h_reference_raw': npz['h_reference_raw'],
                'timestamp': npz['timestamp'],
            }
        except Exception as e:
            print(f"[sfcw] Error loading per_position data: {e}")
            return None

    def load_calibration_status(self):
        status = {
            'cable_thru': None,
            'free_space': None,
            'per_position': None,
        }

        for mode in ['cable_thru', 'free_space']:
            filepath = os.path.join(CALIBRATION_DIR, f'{mode}.npz')
            if os.path.exists(filepath):
                try:
                    npz = np.load(filepath, allow_pickle=True)
                    ts = float(npz['timestamp'])
                    status[mode] = {'timestamp': ts}
                except Exception:
                    pass

        filepath = os.path.join(CALIBRATION_DIR, 'per_position.npz')
        if os.path.exists(filepath):
            try:
                npz = np.load(filepath, allow_pickle=True)
                h = npz['h_complex']
                count = h.shape[0] if h.ndim == 2 else 1
                step_size_cm = float(npz['step_size_cm']) if 'step_size_cm' in npz else None
                status['per_position'] = {
                    'count': count,
                    'step_size_cm': step_size_cm,
                }
            except Exception:
                pass

        return status

    def clear_per_position(self):
        filepath = os.path.join(CALIBRATION_DIR, 'per_position.npz')
        if os.path.exists(filepath):
            os.remove(filepath)
            print("[sfcw] Per-position calibration cleared")

    def undo_per_position(self):
        filepath = os.path.join(CALIBRATION_DIR, 'per_position.npz')
        if not os.path.exists(filepath):
            return

        try:
            npz = np.load(filepath, allow_pickle=True)
            h_complex = npz['h_complex']
            if h_complex.ndim < 2 or h_complex.shape[0] <= 1:
                os.remove(filepath)
                print("[sfcw] Per-position calibration cleared (was single position)")
                return

            save_kwargs = {
                'frequencies': npz['frequencies'],
                'h_complex': h_complex[:-1],
                'h_signal_raw': npz['h_signal_raw'][:-1],
                'h_reference_raw': npz['h_reference_raw'][:-1],
                'params': str(npz['params']),
                'timestamp': npz['timestamp'][:-1],
            }
            if 'step_size_cm' in npz:
                save_kwargs['step_size_cm'] = npz['step_size_cm']
            np.savez(filepath, **save_kwargs)
            print(f"[sfcw] Per-position calibration: removed last position ({h_complex.shape[0] - 1} remaining)")
        except Exception as e:
            print(f"[sfcw] Error undoing per-position: {e}")
