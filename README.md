# Echo — direct desktop brain / Raspberry Pi body

Echo is now split into a powerful desktop brain and a lightweight Raspberry Pi body endpoint.

```text
Pi body
  camera / microphone / speaker / INA219 / HC-05 / Arduino
             |
             | direct encrypted WSS connection initiated by the Pi
             | pinned desktop certificate + shared-secret HMAC
             v
Desktop brain
  original Echo behavior / memory / navigation / face recognition / YOLO
  local Faster-Whisper speech recognition
  local Qwen3-14B 4-bit NF4 GPU cognition
```

There is no VPS image relay and no cloud LLM API in the robot-to-brain path.

## Direct Internet requirement

The Pi can be on a phone hotspot because it makes the outbound connection.

The desktop must be directly reachable from the public Internet. Use either:

- a public IPv4 address with TCP port forwarding from the home router to the desktop; or
- a reachable public IPv6 address with the desktop firewall allowing the configured port.

If the home ISP uses CGNAT and supplies no reachable public IPv4/IPv6 address, a genuinely direct no-middleman connection cannot cross that NAT. In that case request a public/reachable address from the ISP.

## Desktop setup

The default model folder matches the Qwen3 model path used by the supplied researcher script:

```text
C:\Users\garag\OneDrive\Desktop\MME\qwen3_14b
```

Install:

```bash
python setup.py --role desktop --mode install --open-firewall
```

This downloads Echo's vision model files, installs the local-model/runtime dependencies, generates a TLS keypair, writes `echo_desktop_config.json`, and writes `echo_pi_config.generated.json`.

Copy `echo_pi_config.generated.json` to the Pi as `echo_pi_config.json`, then set only `brain_host` to the home's public IPv4/public IPv6/direct DNS name.

Start the brain:

```bash
python echo_desktop.py
```

## Raspberry Pi setup

```bash
python3 setup.py --role pi --mode install
```

Pair/trust the HC-05 as before. Put `echo_pi_config.json` beside the scripts and run:

```bash
python3 main.py
```

`main.py` intentionally remains the Pi launcher so existing startup habits still work.

## What stays on the Pi

- Picamera2 capture and JPEG streaming
- microphone capture and PCM streaming
- speaker playback
- Bluetooth HC-05 / Arduino byte protocol
- INA219/I2C access
- connection/reconnection
- immediate motor watchdog/fail-safe

The Arduino sketch is unchanged. The existing `w/s/a/d/x`, `1`–`5`, and `l` protocol remains intact.

## What moved to the desktop

The full old Echo behavior loop is preserved in `echo_desktop.py`, including its memory/history system, face recognition, YOLO processing, navigation modes, conversation rules, sleep mode, summaries, response correctness review, battery/charger rules, grabber/camera state, and command handling.

Hardware-facing objects are compatibility proxies, so the old code still talks to camera/PyAudio/SMBus/Bluetooth-style objects while those operations execute physically on the Pi.

Cloud OpenAI calls are replaced by local Qwen3. Google speech recognition is replaced by local Faster-Whisper.

## Qwen3 thinking behavior

Normal Echo decisions use Qwen3 with thinking disabled.

The main command chooser now also has:

```text
Think Harder And Then Choose A Command ~~ low
Think Harder And Then Choose A Command ~~ medium
Think Harder And Then Choose A Command ~~ high
Think Harder And Then Choose A Command ~~ very high
```

Only when Echo itself selects one of those does the runtime invoke a second Qwen3 generation with thinking enabled.

The selected effort controls the maximum thinking budget:

- low: 512 tokens
- medium: 1024 tokens
- high: 2048 tokens
- very high: 4096 tokens

If Qwen finishes the thought and final command before that budget, it stops normally; the runtime does not force an unnecessary second continuation. If the budget is exhausted inside the thought, the runtime closes the thought and asks Qwen for the actual executable command.

Sleep summaries, memory-keyword extraction, speech confirmation, and session-response criticism stay in non-thinking mode.

## Security

Generated secrets are intentionally ignored by Git:

- `echo_desktop_config.json`
- `echo_pi_config.json`
- `echo_pi_config.generated.json`
- `echo_cert.pem`
- `echo_key.pem`

The Pi pins the desktop certificate by SHA-256 fingerprint and also authenticates with an HMAC challenge. Do not publish generated configs or keys.

## Files

- `echo_desktop.py` — complete desktop Echo program, transformed from the original monolith
- `echo_qwen.py` — Qwen3-14B 4-bit local model + optional self-selected thinking pass
- `echo_link.py` — encrypted direct transport + desktop hardware compatibility proxies
- `echo_pi.py` — Pi body endpoint
- `main.py` — Pi launcher
- `setup.py` — desktop/Pi setup and TLS/config generator
- `image_server.py` — retired legacy entry point
- `arduino_robot_code.ino` — unchanged
