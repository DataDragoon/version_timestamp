"""bladeRF hardware abstraction — supports dual TX/RX for SFCW reference channel."""

import os
import threading
import time
import numpy as np
import bladerf
from bladerf._bladerf import ChannelLayout, Format, ffi, libbladeRF

SCALE = 2047
MGC = libbladeRF.BLADERF_GAIN_MGC
TUNING_MODE_FPGA = libbladeRF.BLADERF_TUNING_MODE_FPGA

# Metadata-mode constants, values from libbladeRF.h (the cffi binding does not
# export #define macros).
META_FLAG_TX_BURST_START = 1 << 0
META_FLAG_TX_NOW = 1 << 2
META_FLAG_RX_NOW = 1 << 31
META_STATUS_OVERRUN = 1 << 0

# Escape hatch: SFCW_META=0 runs both dual streams in plain SC16_Q11 (no
# in-band timestamps; bladerf_get_timestamp still works with the patched
# FPGA, whose counter runs whenever rx_enable is set). Default: meta on.
# NOTE: metadata must be enabled on BOTH directions or neither — libbladeRF
# rejects mixed formats with BLADERF_ERR_INVAL (the "Invalid operation or
# parameter" sweep error).
META_ENABLED = os.environ.get('SFCW_META', '1') != '0'


class BladeRFDriver:
    def __init__(self):
        self.device = None
        self.tx_running = False
        self.rx_running = False
        self.center_freq = 2_000_000_000
        self.sample_rate = 10_000_000
        self.bandwidth = 1_500_000
        self.tx_gain = 50
        self.rx_gain = 25
        self.tx2_gain = 10
        self.rx2_gain = 0
        self.waveform_type = 'cw'
        self.cw_offset = 100_000
        self.tx_amplitude = 1.0
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
        # Timestamp decode state (populated by the dual RX loop)
        self.last_rx_timestamp = None
        self.rx_first_timestamp = None
        self.rx_buffer_count = 0
        self.rx_ts_step = None
        self.rx_gap_events = 0
        self.rx_lost_samples = 0
        self.rx_corrupt_headers = 0
        self.rx_overrun_events = 0

    def open(self):
        self.device = bladerf.BladeRF()
        self.serial = self.device.get_serial()
        self._configure_channels()

    def close(self):
        self.stop_tx()
        self.stop_rx()
        if self.device:
            self.device.close()
            self.device = None

    def reset(self):
        """Full device close + reopen. Clears all USB/RFIC state."""
        self.stop_tx()
        self.stop_rx()
        self.stop_tx_dual()
        self.stop_rx_dual()
        if self.device:
            self.device.close()
        self.device = bladerf.BladeRF()
        self.serial = self.device.get_serial()
        self._configure_channels()
        print("[bladerf] Device reset complete")

    def _configure_channels(self):
        ch_tx = self.device.Channel(bladerf.CHANNEL_TX(0))
        ch_rx = self.device.Channel(bladerf.CHANNEL_RX(0))
        ch_rx.gain_mode = MGC
        ch_tx.frequency = int(self.center_freq)
        ch_tx.sample_rate = int(self.sample_rate)
        self._warn_if_rate_snapped('TX0', ch_tx.sample_rate)
        ch_tx.bandwidth = int(self.bandwidth)
        ch_tx.gain = int(self.tx_gain)
        ch_rx.frequency = int(self.center_freq)
        ch_rx.sample_rate = int(self.sample_rate)
        self._warn_if_rate_snapped('RX0', ch_rx.sample_rate)
        ch_rx.bandwidth = int(self.bandwidth)
        ch_rx.gain = int(self.rx_gain)

    def _configure_channels_dual(self):
        """Configure all 4 channels (TX1+TX2, RX1+RX2) for SFCW reference mode."""
        dev_ptr = self.device.dev[0]
        gains_tx = [int(self.tx_gain), int(self.tx2_gain)]
        gains_rx = [int(self.rx_gain), int(self.rx2_gain)]
        actual_rate = ffi.new('unsigned int *')

        for ch_idx in range(2):
            tx_ch = bladerf.CHANNEL_TX(ch_idx)
            rx_ch = bladerf.CHANNEL_RX(ch_idx)
            libbladeRF.bladerf_set_frequency(dev_ptr, tx_ch, int(self.center_freq))
            libbladeRF.bladerf_set_sample_rate(dev_ptr, tx_ch, int(self.sample_rate), actual_rate)
            self._warn_if_rate_snapped(f'TX{ch_idx}', actual_rate[0])
            libbladeRF.bladerf_set_bandwidth(dev_ptr, tx_ch, int(self.bandwidth), ffi.NULL)
            libbladeRF.bladerf_set_frequency(dev_ptr, rx_ch, int(self.center_freq))
            libbladeRF.bladerf_set_sample_rate(dev_ptr, rx_ch, int(self.sample_rate), actual_rate)
            self._warn_if_rate_snapped(f'RX{ch_idx}', actual_rate[0])
            libbladeRF.bladerf_set_bandwidth(dev_ptr, rx_ch, int(self.bandwidth), ffi.NULL)
            libbladeRF.bladerf_set_gain_mode(dev_ptr, rx_ch, MGC)
            libbladeRF.bladerf_set_gain(dev_ptr, rx_ch, gains_rx[ch_idx])
            libbladeRF.bladerf_set_gain(dev_ptr, tx_ch, gains_tx[ch_idx])

        print(f"[bladerf] Dual-channel configured: TX1={gains_tx[0]}dB TX2={gains_tx[1]}dB RX1={gains_rx[0]}dB RX2={gains_rx[1]}dB")

    def _warn_if_rate_snapped(self, label, actual_hz):
        """bladerf_set_sample_rate can silently round the requested rate to the
        nearest one the RFIC's clock/decimation chain can actually produce —
        the call succeeds either way. Surface it instead of discarding `actual`,
        since a snapped rate is exactly what drives the total-throughput warning
        libbladeRF logs (it sums the *actual* per-channel rate, not the requested one).
        """
        if actual_hz != int(self.sample_rate):
            print(f"[bladerf] NOTE: {label} sample rate snapped to {actual_hz/1e6:g} Msps "
                  f"(requested {self.sample_rate/1e6:g} Msps)")

    def reapply_dual_gains(self):
        """Re-push TX1/TX2/RX1/RX2 gains after enabling TX/RX modules.

        enable_module() resets gain state, so any dual-channel start (start_tx_dual,
        start_rx_dual) needs this called afterward or the gains configured by
        _configure_channels_dual() are silently lost. Safe to call even if only one
        direction's modules are enabled — setting a gain register for a disabled
        module just takes effect whenever it's next enabled.
        """
        dev_ptr = self.device.dev[0]
        libbladeRF.bladerf_set_gain_mode(dev_ptr, bladerf.CHANNEL_RX(0), MGC)
        libbladeRF.bladerf_set_gain_mode(dev_ptr, bladerf.CHANNEL_RX(1), MGC)
        libbladeRF.bladerf_set_gain(dev_ptr, bladerf.CHANNEL_RX(0), int(self.rx_gain))
        libbladeRF.bladerf_set_gain(dev_ptr, bladerf.CHANNEL_RX(1), int(self.rx2_gain))
        libbladeRF.bladerf_set_gain(dev_ptr, bladerf.CHANNEL_TX(0), int(self.tx_gain))
        libbladeRF.bladerf_set_gain(dev_ptr, bladerf.CHANNEL_TX(1), int(self.tx2_gain))

    def set_tuning_mode_fpga(self):
        """Switch to FPGA tuning mode — retunes execute on-FPGA, no USB round-trip."""
        dev_ptr = self.device.dev[0]
        rc = libbladeRF.bladerf_set_tuning_mode(dev_ptr, TUNING_MODE_FPGA)
        if rc != 0:
            print(f"[bladerf] WARNING: set_tuning_mode FPGA returned {rc}")
        else:
            print("[bladerf] Tuning mode set to FPGA")

    def get_timestamp(self, direction):
        """Get current hardware timestamp (in sample counts) for TX or RX direction."""
        dev_ptr = self.device.dev[0]
        ts = ffi.new('uint64_t *')
        rc = libbladeRF.bladerf_get_timestamp(dev_ptr, direction, ts)
        if rc != 0:
            raise RuntimeError(f"bladerf_get_timestamp failed: rc={rc}")
        return ts[0]

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
            self._warn_if_rate_snapped('TX0', ch_tx.sample_rate)
            ch_tx.bandwidth = self.bandwidth
            ch_rx.sample_rate = self.sample_rate
            self._warn_if_rate_snapped('RX0', ch_rx.sample_rate)
            ch_rx.bandwidth = self.bandwidth
            if self._dual_channel:
                ch_tx2 = self.device.Channel(bladerf.CHANNEL_TX(1))
                ch_rx2 = self.device.Channel(bladerf.CHANNEL_RX(1))
                ch_tx2.sample_rate = self.sample_rate
                self._warn_if_rate_snapped('TX1', ch_tx2.sample_rate)
                ch_tx2.bandwidth = self.bandwidth
                ch_rx2.sample_rate = self.sample_rate
                self._warn_if_rate_snapped('RX1', ch_rx2.sample_rate)
                ch_rx2.bandwidth = self.bandwidth
            self._tx_buffer = self._generate(int(self.sample_rate * 0.01))
            if self._dual_channel:
                self._rebuild_tx_dual_buffer()

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
            if self._dual_channel:
                self._rebuild_tx_dual_buffer()

    def _rebuild_tx_dual_buffer(self):
        """Rebuild the interleaved TX1+TX2 buffer from self._tx_buffer.

        Call whenever self._tx_buffer changes while dual TX may be running —
        _tx_loop_dual re-reads _tx_dual_bytes every iteration (like _tx_loop does
        with _tx_buffer), so this is what makes live waveform/rate changes actually
        reach a running dual-channel TX instead of silently doing nothing.
        """
        buf = self._tx_buffer
        n_samples = len(buf) // 2
        tx_dual_buf = np.empty(len(buf) * 2, dtype=np.int16)
        tx_dual_buf[0::4] = buf[0::2]  # TX1 I
        tx_dual_buf[1::4] = buf[1::2]  # TX1 Q
        tx_dual_buf[2::4] = buf[0::2]  # TX2 I
        tx_dual_buf[3::4] = buf[1::2]  # TX2 Q
        self._tx_dual_buf = tx_dual_buf
        self._tx_dual_bytes = tx_dual_buf.tobytes()
        self._tx_dual_n_samples = n_samples

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
        self._tx_buffer = self._generate(int(self.sample_rate * 0.01))
        self._tx_stop.clear()
        self.tx_running = True
        self.device.sync_config(
            layout=ChannelLayout.TX_X1,
            fmt=Format.SC16_Q11,
            num_buffers=16,
            buffer_size=4096,
            num_transfers=8,
            stream_timeout=3500
        )
        self.device.enable_module(bladerf.CHANNEL_TX(0), True)
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
        self._rx_stop.clear()
        self.rx_running = True
        self.device.sync_config(
            layout=ChannelLayout.RX_X1,
            fmt=Format.SC16_Q11,
            num_buffers=16,
            buffer_size=4096,
            num_transfers=8,
            stream_timeout=3500
        )
        self.device.enable_module(bladerf.CHANNEL_RX(0), True)
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

    def start_tx_dual(self):
        """Start TX on both channels (TX1=antenna, TX2=reference cable)."""
        if self.tx_running:
            return
        self._tx_buffer = self._generate(int(self.sample_rate * 0.01))
        self._tx_stop.clear()
        self.tx_running = True
        self._dual_channel = True
        self._rebuild_tx_dual_buffer()
        self.device.sync_config(
            layout=ChannelLayout.TX_X2,
            fmt=Format.SC16_Q11_META if META_ENABLED else Format.SC16_Q11,
            num_buffers=16,
            buffer_size=4096,
            num_transfers=8,
            stream_timeout=3500
        )
        self.device.enable_module(bladerf.CHANNEL_TX(0), True)
        self.device.enable_module(bladerf.CHANNEL_TX(1), True)
        self._tx_thread = threading.Thread(target=self._tx_loop_dual, daemon=True)
        self._tx_thread.start()

    def _tx_loop_dual(self):
        """TX loop for dual channel — replays the interleaved buffer, re-read each
        iteration (like _tx_loop) so live waveform/rate changes take effect.

        Metadata mode: the first call opens one continuous burst (TX_NOW +
        BURST_START); every later call passes flags=0, appending to the same
        burst with contiguous timestamps so the CW tone stays seamless."""
        dev_ptr = self.device.dev[0]
        tx_meta = None
        if META_ENABLED:
            tx_meta = ffi.new('struct bladerf_metadata *')
            tx_meta.flags = META_FLAG_TX_NOW | META_FLAG_TX_BURST_START
        try:
            while not self._tx_stop.is_set():
                with self._lock:
                    tx_bytes = self._tx_dual_bytes
                    n_samples = self._tx_dual_n_samples
                if tx_meta is not None:
                    rc = libbladeRF.bladerf_sync_tx(dev_ptr, ffi.from_buffer(tx_bytes),
                                                    n_samples, tx_meta, 3500)
                    if rc != 0:
                        print(f"[bladerf] sync_tx(meta) error: rc={rc}", flush=True)
                        break
                    tx_meta.flags = 0  # continue the burst
                else:
                    self.device.sync_tx(tx_bytes, n_samples)
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
        self._rx_stop.clear()
        self.rx_running = True
        self._dual_channel = True
        # Timestamp decode state (read by the SFCW engine)
        self.last_rx_timestamp = None
        self.rx_first_timestamp = None
        self.rx_buffer_count = 0
        self.rx_ts_step = None      # learned counter advance per buffer
        self.rx_gap_events = 0
        self.rx_lost_samples = 0
        self.rx_corrupt_headers = 0
        self.rx_overrun_events = 0
        print(f"[bladerf] RX dual stream: "
              f"{'SC16_Q11_META (in-band timestamps)' if META_ENABLED else 'SC16_Q11 (plain, SFCW_META=0)'}",
              flush=True)
        # USB link speed decides the metadata message size: SuperSpeed = 8192-byte
        # messages, HighSpeed = 2048. A host/FPGA disagreement here misaligns
        # header parsing (the 1-in-4 sane timestamp symptom).
        try:
            speed = libbladeRF.bladerf_device_speed(self.device.dev[0])
            name, msg = {
                libbladeRF.BLADERF_DEVICE_SPEED_SUPER: ("SuperSpeed (USB 3.0)", 8192),
                libbladeRF.BLADERF_DEVICE_SPEED_HIGH: ("HighSpeed (USB 2.0)", 2048),
            }.get(speed, (f"UNKNOWN ({speed})", 0))
            print(f"[bladerf] USB link: {name} -> {msg}-byte metadata messages",
                  flush=True)
        except Exception as e:
            print(f"[bladerf] USB speed query failed: {e}", flush=True)
        self.device.sync_config(
            layout=ChannelLayout.RX_X2,
            fmt=Format.SC16_Q11_META if META_ENABLED else Format.SC16_Q11,
            num_buffers=16,
            buffer_size=4096,
            num_transfers=8,
            stream_timeout=3500
        )
        self.device.enable_module(bladerf.CHANNEL_RX(0), True)
        self.device.enable_module(bladerf.CHANNEL_RX(1), True)
        self._rx_thread = threading.Thread(target=self._rx_loop_dual, args=(callback, num_samples), daemon=True)
        self._rx_thread.start()

    def _rx_loop_dual(self, callback, num_samples):
        """RX loop for dual channel — deinterleaves RX1 and RX2."""
        # RX_X2: interleaved [RX1_I, RX1_Q, RX2_I, RX2_Q, ...]
        # num_samples is per-channel, so total buffer is num_samples * 2 channels * 2 (I+Q) * 2 bytes
        buf = bytearray(num_samples * 2 * 2 * 2)
        dev_ptr = self.device.dev[0]
        meta = ffi.new('struct bladerf_metadata *') if META_ENABLED else None
        prev_ts = None
        try:
            while not self._rx_stop.is_set():
                if meta is None:
                    # Plain mode: no in-band timestamps to decode
                    self.device.sync_rx(buf, num_samples)
                else:
                    # RX_NOW = stream continuously and report each buffer's
                    # timestamp (flags=0 would mean "capture at meta.timestamp"
                    # and fail with BLADERF_ERR_TIME_PAST).
                    meta.flags = META_FLAG_RX_NOW
                    rc = libbladeRF.bladerf_sync_rx(dev_ptr, ffi.from_buffer(buf),
                                                    num_samples, meta, 3500)
                    if rc != 0:
                        print(f"[bladerf] sync_rx(meta) error: rc={rc}", flush=True)
                        break

                    ts = meta.timestamp
                    self.rx_buffer_count += 1

                    # A header misparse (message-size mismatch, corrupt meta
                    # path) shows up as absurd 64-bit values. Classify those
                    # separately so they don't poison the gap statistics.
                    corrupt = ts > (1 << 48)  # > ~2.8e14: >8.9 years of samples
                    if corrupt:
                        self.rx_corrupt_headers += 1
                        if self.rx_corrupt_headers <= 3 or self.rx_corrupt_headers % 1000 == 0:
                            print(f"[bladerf] CORRUPT header #{self.rx_corrupt_headers}: "
                                  f"ts={ts} (sample data misread as header?)",
                                  flush=True)
                    else:
                        self.last_rx_timestamp = ts
                        if self.rx_first_timestamp is None:
                            self.rx_first_timestamp = ts
                            print(f"[bladerf] FIRST RX timestamp: {ts} "
                                  f"(FPGA sample counter at this buffer's first sample)",
                                  flush=True)
                        elif prev_ts is not None:
                            gap = ts - prev_ts
                            if self.rx_ts_step is None and gap > 0:
                                # Learn the nominal per-buffer advance from the
                                # first clean gap instead of assuming layout math.
                                self.rx_ts_step = gap
                                print(f"[bladerf] RX timestamp advance per buffer: "
                                      f"{gap} counter ticks", flush=True)
                            elif self.rx_ts_step and gap != self.rx_ts_step:
                                self.rx_gap_events += 1
                                self.rx_lost_samples += max(0, gap - self.rx_ts_step)
                                # Rate-limit: first 5 gaps, then every 1000th
                                if self.rx_gap_events <= 5 or self.rx_gap_events % 1000 == 0:
                                    print(f"[bladerf] RX TIMESTAMP GAP #{self.rx_gap_events}: "
                                          f"prev={prev_ts} now={ts} gap={gap} "
                                          f"(expected {self.rx_ts_step})", flush=True)
                        prev_ts = ts

                    if meta.status & META_STATUS_OVERRUN:
                        self.rx_overrun_events += 1
                        # Rate-limit: first 5, then every 1000th
                        if self.rx_overrun_events <= 5 or self.rx_overrun_events % 1000 == 0:
                            print(f"[bladerf] RX OVERRUN #{self.rx_overrun_events} "
                                  f"at timestamp {ts}", flush=True)

                iq = np.frombuffer(buf, dtype=np.int16).copy()
                # Deinterleave: [I1, Q1, I2, Q2, I1, Q1, I2, Q2, ...]
                rx1 = np.empty(num_samples * 2, dtype=np.int16)
                rx2 = np.empty(num_samples * 2, dtype=np.int16)
                rx1[0::2] = iq[0::4]  # RX1 I
                rx1[1::2] = iq[1::4]  # RX1 Q
                rx2[0::2] = iq[2::4]  # RX2 I
                rx2[1::2] = iq[3::4]  # RX2 Q
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
        # Decode summary: buffers received vs what the FPGA counter says elapsed
        if (self.rx_first_timestamp is not None
                and self.last_rx_timestamp is not None
                and self.rx_ts_step):
            span = self.last_rx_timestamp - self.rx_first_timestamp + self.rx_ts_step
            expected = self.rx_buffer_count * self.rx_ts_step
            print(f"[bladerf] RX stream totals: buffers={self.rx_buffer_count} "
                  f"counter_span={span} expected={expected} "
                  f"gaps={self.rx_gap_events} lost={self.rx_lost_samples} ticks "
                  f"corrupt_headers={self.rx_corrupt_headers} "
                  f"overruns={self.rx_overrun_events}",
                  flush=True)

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
