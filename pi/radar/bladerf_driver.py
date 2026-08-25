"""bladeRF hardware abstraction — supports dual TX/RX for SFCW reference channel."""

import time
import threading
import numpy as np
import bladerf
from bladerf._bladerf import ChannelLayout, Format, ffi, libbladeRF

SCALE = 2047
MGC = libbladeRF.BLADERF_GAIN_MGC

# AD9361 RFIC registers for tracking calibration control
_REG_CAL_CONFIG_2 = 0x16A  # Bit 0: BBDC tracking, Bit 1: RFDC tracking
_REG_CAL_CONFIG_3 = 0x16B  # Bit 0: RX Quadrature tracking


class BladeRFDriver:
    def __init__(self):
        self.device = None
        self.tx_running = False
        self.rx_running = False
        self.center_freq = 915_000_000
        self.sample_rate = 2_000_000
        self.bandwidth = 1_500_000
        self.tx_gain = 47
        self.rx_gain = 30
        self.tx2_gain = 10
        self.rx2_gain = 0
        self.waveform_type = 'cw'
        self.cw_offset = 100_000
        self.tx_amplitude = 0.8
        self.chirp_bw = 500_000
        self.chirp_duration = 0.001
        self.serial = None
        self._tx_thread = None
        self._rx_thread = None
        self._tx_stop = threading.Event()
        self._rx_stop = threading.Event()
        self._lock = threading.Lock()
        self._tx_buffer = None
        self._dual_channel = False
        self._last_layout = None  # 'x1' or 'x2' — tracks sync_config state

    def open(self):
        self.device = bladerf.BladeRF()
        self.serial = self.device.get_serial()
        self._configure_channels()
        self._lock_tracking_calibrations()

    def close(self):
        self.stop_tx()
        self.stop_rx()
        if self.device:
            self.device.close()
            self.device = None

    def reopen(self):
        """Close and reopen device — required to reset sync_config channel layout."""
        if self.device:
            self.device.close()
        self.device = bladerf.BladeRF()
        self.serial = self.device.get_serial()
        self._dual_channel = False
        self._last_layout = None
        self._lock_tracking_calibrations()

    def _lock_tracking_calibrations(self):
        """Disable AD9361 continuous tracking calibrations for deterministic gain.

        The AD9361 runs background loops that continuously adjust RX quadrature,
        BB DC offset, and RF DC offset corrections. These cause non-deterministic
        amplitude/phase variation even with fixed gain registers. We run one-shot
        calibration at init, then freeze the correction coefficients.
        """
        try:
            # Trigger one-shot RX quadrature calibration before freezing.
            # AD9361 reg 0x016 (Calibration Control): writing 0x32 triggers
            # RX quad cal + TX quad cal (one-shot, not tracking).
            self._write_rfic_reg(0x016, 0x32)
            time.sleep(0.1)  # Wait for cal to complete

            # Disable RX quadrature tracking (reg 0x16B bit 0)
            reg_val = self._read_rfic_reg(_REG_CAL_CONFIG_3)
            reg_val &= ~0x01
            self._write_rfic_reg(_REG_CAL_CONFIG_3, reg_val)

            # Disable BB DC tracking and RF DC tracking (reg 0x16A bits 0,1)
            reg_val = self._read_rfic_reg(_REG_CAL_CONFIG_2)
            reg_val &= ~0x03
            self._write_rfic_reg(_REG_CAL_CONFIG_2, reg_val)

            print("[bladerf] Tracking calibrations locked (RX quad, BBDC, RFDC disabled)")
        except RuntimeError as e:
            print(f"[bladerf] WARNING: Could not lock tracking cals: {e}")
            print("[bladerf]   Non-deterministic gain behavior may persist")

    def run_oneshot_calibration(self):
        """Trigger one-shot RX/TX quadrature calibration without enabling tracking.

        Call this before a sweep begins (at the sweep's center frequency) to get
        fresh IQ correction coefficients. Tracking remains disabled afterward.
        """
        try:
            self._write_rfic_reg(0x016, 0x32)
            time.sleep(0.1)
        except RuntimeError:
            pass

    def _read_rfic_reg(self, addr):
        """Read AD9361 register via libbladeRF RFIC SPI interface."""
        dev_ptr = self.device.dev[0]
        val = ffi.new("uint8_t *")
        rc = libbladeRF.bladerf_get_rfic_register(dev_ptr, 0, int(addr), val)
        if rc != 0:
            raise RuntimeError(f"RFIC read reg 0x{addr:03X} failed: {rc}")
        return val[0]

    def _write_rfic_reg(self, addr, val):
        """Write AD9361 register via libbladeRF RFIC SPI interface."""
        dev_ptr = self.device.dev[0]
        rc = libbladeRF.bladerf_set_rfic_register(dev_ptr, 0, int(addr), int(val) & 0xFF)
        if rc != 0:
            raise RuntimeError(f"RFIC write reg 0x{addr:03X}=0x{val:02X} failed: {rc}")

    def _configure_channels(self):
        ch_tx = self.device.Channel(bladerf.CHANNEL_TX(0))
        ch_rx = self.device.Channel(bladerf.CHANNEL_RX(0))
        ch_rx.gain_mode = MGC
        ch_tx.frequency = int(self.center_freq)
        ch_tx.sample_rate = int(self.sample_rate)
        ch_tx.bandwidth = int(self.bandwidth)
        ch_tx.gain = int(self.tx_gain)
        ch_rx.frequency = int(self.center_freq)
        ch_rx.sample_rate = int(self.sample_rate)
        ch_rx.bandwidth = int(self.bandwidth)
        ch_rx.gain = int(self.rx_gain)

    def _configure_channels_dual(self):
        """Configure all 4 channels (TX1+TX2, RX1+RX2) for SFCW reference mode."""
        dev_ptr = self.device.dev[0]
        gains_tx = [int(self.tx_gain), int(self.tx2_gain)]
        gains_rx = [int(self.rx_gain), int(self.rx2_gain)]

        for ch_idx in range(2):
            tx_ch = bladerf.CHANNEL_TX(ch_idx)
            rx_ch = bladerf.CHANNEL_RX(ch_idx)
            libbladeRF.bladerf_set_frequency(dev_ptr, tx_ch, int(self.center_freq))
            libbladeRF.bladerf_set_sample_rate(dev_ptr, tx_ch, int(self.sample_rate), ffi.NULL)
            libbladeRF.bladerf_set_bandwidth(dev_ptr, tx_ch, int(self.bandwidth), ffi.NULL)
            libbladeRF.bladerf_set_frequency(dev_ptr, rx_ch, int(self.center_freq))
            libbladeRF.bladerf_set_sample_rate(dev_ptr, rx_ch, int(self.sample_rate), ffi.NULL)
            libbladeRF.bladerf_set_bandwidth(dev_ptr, rx_ch, int(self.bandwidth), ffi.NULL)
            libbladeRF.bladerf_set_gain_mode(dev_ptr, rx_ch, MGC)
            libbladeRF.bladerf_set_gain(dev_ptr, rx_ch, gains_rx[ch_idx])
            libbladeRF.bladerf_set_gain(dev_ptr, tx_ch, gains_tx[ch_idx])

        print(f"[bladerf] Dual-channel configured: TX1={gains_tx[0]}dB TX2={gains_tx[1]}dB RX1={gains_rx[0]}dB RX2={gains_rx[1]}dB")

    def set_frequency(self, freq_hz):
        with self._lock:
            self.center_freq = int(freq_hz)
            self.device.Channel(bladerf.CHANNEL_TX(0)).frequency = self.center_freq
            self.device.Channel(bladerf.CHANNEL_RX(0)).frequency = self.center_freq
            if self._dual_channel:
                self.device.Channel(bladerf.CHANNEL_TX(1)).frequency = self.center_freq
                self.device.Channel(bladerf.CHANNEL_RX(1)).frequency = self.center_freq

    def set_tx_gain(self, gain_db):
        with self._lock:
            self.tx_gain = int(gain_db)
            self.device.Channel(bladerf.CHANNEL_TX(0)).gain = self.tx_gain

    def set_rx_gain(self, gain_db):
        with self._lock:
            self.rx_gain = int(gain_db)
            ch = self.device.Channel(bladerf.CHANNEL_RX(0))
            ch.gain_mode = MGC
            ch.gain = self.rx_gain

    def set_sample_rate(self, rate):
        with self._lock:
            self.sample_rate = int(rate)
            self.bandwidth = int(rate * 0.75)
            ch_tx = self.device.Channel(bladerf.CHANNEL_TX(0))
            ch_rx = self.device.Channel(bladerf.CHANNEL_RX(0))
            ch_tx.sample_rate = self.sample_rate
            ch_tx.bandwidth = self.bandwidth
            ch_rx.sample_rate = self.sample_rate
            ch_rx.bandwidth = self.bandwidth
            self._tx_buffer = self._generate(int(self.sample_rate * 0.01))

    def set_waveform(self, waveform_type, **params):
        with self._lock:
            self.waveform_type = waveform_type
            if 'offset' in params:
                self.cw_offset = int(params['offset'])
            if 'amplitude' in params:
                self.tx_amplitude = float(params['amplitude'])
            if 'chirp_bw' in params:
                self.chirp_bw = int(params['chirp_bw'])
            if 'chirp_duration' in params:
                self.chirp_duration = float(params['chirp_duration'])
            self._tx_buffer = self._generate(int(self.sample_rate * 0.01))

    def _generate(self, num_samples):
        if self.waveform_type == 'chirp':
            return self._gen_chirp(num_samples)
        elif self.waveform_type == 'noise':
            return self._gen_noise(num_samples)
        return self._gen_cw(num_samples)

    def _gen_cw(self, n):
        t = np.arange(n, dtype=np.float64) / self.sample_rate
        phase = 2 * np.pi * self.cw_offset * t
        iq = np.empty(n * 2, dtype=np.int16)
        iq[0::2] = np.clip(np.cos(phase) * self.tx_amplitude * SCALE, -2048, 2047).astype(np.int16)
        iq[1::2] = np.clip(np.sin(phase) * self.tx_amplitude * SCALE, -2048, 2047).astype(np.int16)
        return iq

    def _gen_chirp(self, n):
        t = np.arange(n, dtype=np.float64) / self.sample_rate
        f0 = -self.chirp_bw / 2
        f1 = self.chirp_bw / 2
        t_mod = t % self.chirp_duration
        phase = 2 * np.pi * (f0 * t_mod + (f1 - f0) / (2 * self.chirp_duration) * t_mod ** 2)
        iq = np.empty(n * 2, dtype=np.int16)
        iq[0::2] = np.clip(np.cos(phase) * self.tx_amplitude * SCALE, -2048, 2047).astype(np.int16)
        iq[1::2] = np.clip(np.sin(phase) * self.tx_amplitude * SCALE, -2048, 2047).astype(np.int16)
        return iq

    def _gen_noise(self, n):
        noise = np.random.randn(n * 2) * self.tx_amplitude * SCALE * 0.5
        return np.clip(noise, -2048, 2047).astype(np.int16)

    # -- Single-channel TX/RX (used by RF Calib panel) --

    def start_tx(self):
        if self.tx_running:
            return
        if self._last_layout == 'x2':
            self.reopen()
        self._tx_buffer = self._generate(int(self.sample_rate * 0.01))
        self._tx_stop.clear()
        self.tx_running = True
        self.device.enable_module(bladerf.CHANNEL_TX(0), True)
        self.device.sync_config(
            layout=ChannelLayout.TX_X1,
            fmt=Format.SC16_Q11,
            num_buffers=16,
            buffer_size=4096,
            num_transfers=8,
            stream_timeout=3500
        )
        self._last_layout = 'x1'
        self._tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
        self._tx_thread.start()

    def _tx_loop(self):
        try:
            while not self._tx_stop.is_set():
                with self._lock:
                    buf = self._tx_buffer
                self.device.sync_tx(buf.tobytes(), len(buf) // 2)
        except Exception as e:
            print(f"[bladerf] TX error: {e}")
        finally:
            try:
                self.device.enable_module(bladerf.CHANNEL_TX(0), False)
            except Exception:
                pass
            self.tx_running = False

    def stop_tx(self):
        if not self.tx_running:
            return
        self._tx_stop.set()
        if self._tx_thread:
            self._tx_thread.join(timeout=2)
            self._tx_thread = None
        self.tx_running = False

    def start_rx(self, callback, num_samples=16384):
        if self.rx_running:
            return
        if self._last_layout == 'x2':
            self.reopen()
        self._rx_stop.clear()
        self.rx_running = True
        self.device.enable_module(bladerf.CHANNEL_RX(0), True)
        self.device.sync_config(
            layout=ChannelLayout.RX_X1,
            fmt=Format.SC16_Q11,
            num_buffers=16,
            buffer_size=4096,
            num_transfers=8,
            stream_timeout=3500
        )
        self._last_layout = 'x1'
        self._rx_thread = threading.Thread(target=self._rx_loop, args=(callback, num_samples), daemon=True)
        self._rx_thread.start()

    def _rx_loop(self, callback, num_samples):
        buf = bytearray(num_samples * 2 * 2)
        try:
            while not self._rx_stop.is_set():
                self.device.sync_rx(buf, num_samples)
                iq = np.frombuffer(buf, dtype=np.int16).copy()
                callback(iq)
        except Exception as e:
            print(f"[bladerf] RX error: {e}")
        finally:
            try:
                self.device.enable_module(bladerf.CHANNEL_RX(0), False)
            except Exception:
                pass
            self.rx_running = False

    def stop_rx(self):
        if not self.rx_running:
            return
        self._rx_stop.set()
        if self._rx_thread:
            self._rx_thread.join(timeout=2)
            self._rx_thread = None
        self.rx_running = False

    # -- Dual-channel TX/RX (used by SFCW engine for reference channel) --

    def start_tx_dual(self, tx2_digital_scale=1.0):
        """Start TX on both channels (TX1=antenna, TX2=reference cable).

        tx2_digital_scale: scale factor for TX2 digital samples (0.0-1.0).
        Use <1.0 to reduce TX2 output power without changing analog gain setting,
        preserving phase-matched behavior with TX1 at the same gain register value.
        """
        if self.tx_running:
            return
        if self._last_layout == 'x1':
            self.reopen()
        self._tx_buffer = self._generate(int(self.sample_rate * 0.01))
        self._tx2_digital_scale = float(tx2_digital_scale)
        self._tx_stop.clear()
        self.tx_running = True
        self._dual_channel = True
        self.device.sync_config(
            layout=ChannelLayout.TX_X2,
            fmt=Format.SC16_Q11,
            num_buffers=16,
            buffer_size=4096,
            num_transfers=8,
            stream_timeout=3500
        )
        self._last_layout = 'x2'
        self.device.enable_module(bladerf.CHANNEL_TX(0), True)
        self.device.enable_module(bladerf.CHANNEL_TX(1), True)
        self._tx_thread = threading.Thread(target=self._tx_loop_dual, daemon=True)
        self._tx_thread.start()

    def _tx_loop_dual(self):
        """TX loop for dual channel — interleaved TX1+TX2 samples."""
        try:
            while not self._tx_stop.is_set():
                with self._lock:
                    buf = self._tx_buffer
                    scale2 = self._tx2_digital_scale
                n_samples = len(buf) // 2
                dual_buf = np.empty(len(buf) * 2, dtype=np.int16)
                dual_buf[0::4] = buf[0::2]  # TX1 I
                dual_buf[1::4] = buf[1::2]  # TX1 Q
                if scale2 >= 1.0:
                    dual_buf[2::4] = buf[0::2]  # TX2 I
                    dual_buf[3::4] = buf[1::2]  # TX2 Q
                else:
                    dual_buf[2::4] = (buf[0::2].astype(np.float64) * scale2).astype(np.int16)
                    dual_buf[3::4] = (buf[1::2].astype(np.float64) * scale2).astype(np.int16)
                self.device.sync_tx(dual_buf.tobytes(), n_samples)
        except Exception as e:
            print(f"[bladerf] TX dual error: {e}")
        finally:
            try:
                self.device.enable_module(bladerf.CHANNEL_TX(0), False)
                self.device.enable_module(bladerf.CHANNEL_TX(1), False)
            except Exception:
                pass
            self.tx_running = False

    def stop_tx_dual(self):
        if not self.tx_running:
            return
        self._tx_stop.set()
        if self._tx_thread:
            self._tx_thread.join(timeout=2)
            self._tx_thread = None
        self.tx_running = False
        self._dual_channel = False

    def start_rx_dual(self, callback, num_samples=1024):
        """Start RX on both channels. Callback receives (rx1_iq, rx2_iq) tuple."""
        if self.rx_running:
            return
        if self._last_layout == 'x1':
            self.reopen()
        self._rx_stop.clear()
        self.rx_running = True
        self._dual_channel = True
        buf_size = max(4096, num_samples * 2)
        self.device.sync_config(
            layout=ChannelLayout.RX_X2,
            fmt=Format.SC16_Q11,
            num_buffers=16,
            buffer_size=buf_size,
            num_transfers=8,
            stream_timeout=3500
        )
        self._last_layout = 'x2'
        self.device.enable_module(bladerf.CHANNEL_RX(0), True)
        self.device.enable_module(bladerf.CHANNEL_RX(1), True)
        self._rx_thread = threading.Thread(target=self._rx_loop_dual, args=(callback, num_samples), daemon=True)
        self._rx_thread.start()

    def _rx_loop_dual(self, callback, num_samples):
        """RX loop for dual channel — deinterleaves RX1 and RX2."""
        # RX_X2: sync_rx(buf, N) captures N sample-pairs total.
        # Each pair = [I1,Q1,I2,Q2] = 4 int16. Yields N samples per channel.
        # Request num_samples*2 to get num_samples per channel after deinterleave.
        rx_count = num_samples * 2
        buf = bytearray(rx_count * 4 * 2)  # rx_count pairs × 4 int16 × 2 bytes
        try:
            while not self._rx_stop.is_set():
                self.device.sync_rx(buf, rx_count)
                iq = np.frombuffer(buf, dtype=np.int16).copy()
                # iq has rx_count*4 int16 values: [I1,Q1,I2,Q2, I1,Q1,I2,Q2, ...]
                # Each channel has rx_count values, but we want num_samples per ch
                rx1 = np.empty(num_samples * 2, dtype=np.int16)
                rx2 = np.empty(num_samples * 2, dtype=np.int16)
                rx1[0::2] = iq[0::4][:num_samples]
                rx1[1::2] = iq[1::4][:num_samples]
                rx2[0::2] = iq[2::4][:num_samples]
                rx2[1::2] = iq[3::4][:num_samples]
                callback(rx1, rx2)
        except Exception as e:
            print(f"[bladerf] RX dual error: {e}")
        finally:
            try:
                self.device.enable_module(bladerf.CHANNEL_RX(0), False)
                self.device.enable_module(bladerf.CHANNEL_RX(1), False)
            except Exception:
                pass
            self.rx_running = False

    def stop_rx_dual(self):
        if not self.rx_running:
            return
        self._rx_stop.set()
        if self._rx_thread:
            self._rx_thread.join(timeout=2)
            self._rx_thread = None
        self.rx_running = False
        self._dual_channel = False

    def get_status(self):
        return {
            'connected': self.device is not None,
            'serial': self.serial,
            'freq': self.center_freq,
            'sample_rate': self.sample_rate,
            'bandwidth': self.bandwidth,
            'tx_gain': self.tx_gain,
            'rx_gain': self.rx_gain,
            'tx_active': self.tx_running,
            'rx_active': self.rx_running,
            'waveform': self.waveform_type,
            'cw_offset': self.cw_offset,
            'tx_amplitude': self.tx_amplitude,
            'chirp_bw': self.chirp_bw,
            'chirp_duration': self.chirp_duration,
        }
