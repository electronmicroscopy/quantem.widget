# Viewing exported HTML on mobile (the WebGPU secure-context rule)

This is the single most important and most confusing fact about opening an
exported interactive widget (`Show4DSTEM`, `Show3D`, `Show3DSlices`, `Show2D`,
`ShowEDS`) on a phone or tablet.

## The rule

Interactive widgets recompute every frame in the browser with **WebGPU**
(probe -> diffraction pattern, detector -> virtual image, contrast, FFT, etc.).
**Browsers expose `navigator.gpu` only in a secure context: HTTPS, or
`localhost`/`127.0.0.1`.**

Therefore an exported HTML opened on a phone **must be served over HTTPS**.
Serving it over plain `http://<LAN-or-Tailscale-IP>:<port>/...` makes
`navigator.gpu` `undefined`, and the widget cannot recompute. It will render the
first frame and the controls, but **taps and drags do nothing** because the GPU
compute path is missing.

Why this fools everyone:

- On a **desktop** it "just works" because the dev URL is
  `http://127.0.0.1:<port>` and **localhost is a secure context**, so WebGPU is
  exposed over plain HTTP there.
- **WebGL is NOT gated** by secure context, so a quick GPU test still shows the
  GPU name and renders. The hardware is obviously present; only the *WebGPU API*
  is withheld. People conclude "the phone has no GPU" or "the widget is broken"
  when the real cause is the insecure origin.

## The fix: serve over HTTPS (pick one)

An exported widget is a single self-contained `.html` file, so any of these
gives it the secure context WebGPU needs. Ordered by audience.

### Option 1: a static HTTPS host (simplest; best for sharing a finished file)

No server, no tunnel. Upload the `.html` to anything that serves HTTPS and open
the link on the phone:

- **Netlify Drop** (`app.netlify.com/drop`): drag the file in, get an instant
  `https://...netlify.app/...` URL.
- **GitHub Pages**: commit the `.html` to a Pages-enabled repo ->
  `https://<user>.github.io/<repo>/<file>.html`.
- Cloudflare Pages, Vercel, S3 + CloudFront, etc.

A large `Show4DSTEM` export can be tens of MB; that is fine for these hosts. This
is the right choice for most people who just want to send someone a working link.

### Option 2: Tailscale (best for private/local, nothing public)

One-time setup:

1. Install Tailscale on the serving machine
   (`curl -fsSL https://tailscale.com/install.sh | sh` then `tailscale up`) and
   the **Tailscale app on the phone** (App Store / Play Store), both signed into
   the **same account** (same tailnet).
2. In the admin console (`login.tailscale.com`): enable **MagicDNS** and
   **HTTPS Certificates** (one toggle each).

Serve the local port over HTTPS:

```bash
tailscale serve --bg --https=443 http://127.0.0.1:<port>
# -> https://<machine>.<tailnet>.ts.net/...   (real Let's Encrypt cert)
# stop with:  tailscale serve --https=443 off
```

`tailscale cert <machine>.<tailnet>.ts.net` succeeding confirms HTTPS is enabled.
Open the `https://` URL on the phone with **Tailscale ON**. It is reachable only
on your tailnet; nothing is exposed to the public internet.

**Verified worked example** (this lab's Linux compute box, Tailscale 1.98.4, file
server on `:8780`). Find your own machine name with `tailscale status` (the
`DNSName` field):

```console
$ tailscale serve --bg --https=443 http://127.0.0.1:8780
Available within your tailnet:
https://owner-ms-7e34.tail9632de.ts.net/
|-- proxy http://127.0.0.1:8780
Serve started and running in the background.

$ tailscale serve status
https://owner-ms-7e34.tail9632de.ts.net (tailnet only)
|-- / proxy http://127.0.0.1:8780
```

The widget then opens on the phone at
`https://owner-ms-7e34.tail9632de.ts.net/agent-show/<file>.html` (HTTP 200, valid
cert, secure context, so `navigator.gpu` is exposed). Substitute your own
`DNSName` and the local port your server listens on.

### Option 3: a quick public tunnel (one-off checks)

```bash
cloudflared tunnel --url http://localhost:<port>   # prints a https://...trycloudflare.com URL
# or
ngrok http <port>                                  # prints a https URL (free account)
```

Open the printed `https://` URL on the phone. Anyone with the URL can reach it
while the tunnel runs, so use it for quick tests, not sensitive data.

### Option 4: localhost (same machine only)

On the serving machine itself, `http://localhost:<port>` is already a secure
context, so WebGPU works there. This does **not** help a phone (a phone is not
localhost).

### What does NOT work

- Plain `http://<LAN-IP>` or `http://<tailscale-IP>` opened on a phone: insecure
  origin, `navigator.gpu` withheld.
- Self-signed HTTPS on iOS Safari: the certificate warning prevents a trusted
  secure context, so WebGPU stays hidden. Use a real cert (Options 1-3).

## Browser notes (iOS)

- **iOS/iPadOS 18 Safari** can do WebGPU, sometimes behind a flag:
  Settings -> Apps -> Safari -> Advanced -> Feature Flags -> **WebGPU = ON**.
- **Brave / Chrome / Firefox on iOS** are forced to use WebKit and (Brave
  especially, via shields) **do not expose WebGPU**. Use the real **Safari** app.
- **Low Power Mode** and **Lockdown Mode** disable WebGPU. Turn both off.
- The Safari flag does nothing over plain HTTP: secure context is required first.

## Debugging: a self-contained diagnostic page

Serve this over the SAME origin as the widget and open it on the phone. It
separates the four independent failure points.

```html
<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<pre id=o style="font:14px monospace;padding:12px">checking...</pre>
<script>(async()=>{
  const L=[];
  L.push("secureContext: "+window.isSecureContext);            // false => served over HTTP, FIX THIS FIRST
  let gl=document.createElement("canvas").getContext("webgl2")||document.createElement("canvas").getContext("webgl");
  if(gl){const d=gl.getExtension("WEBGL_debug_renderer_info");
    L.push("WebGL renderer: "+(d?gl.getParameter(d.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER)));} // names the GPU => hardware present
  else L.push("WebGL: none");
  if(!navigator.gpu) L.push("navigator.gpu: MISSING");          // missing in secure context => Safari flag off / Brave
  else { try{const a=await navigator.gpu.requestAdapter();
    L.push("navigator.gpu: present, adapter="+(a?"OK":"null"));} // null => Low Power/Lockdown/blocked
    catch(e){L.push("requestAdapter error: "+e);} }
  L.push("UA: "+navigator.userAgent);                           // shows Safari vs Brave
  document.getElementById("o").textContent=L.join("\n");
})();</script>
```

### Decision table

| Symptom | Cause | Fix |
| --- | --- | --- |
| `secureContext: false` | Served over plain HTTP to a non-localhost host | Serve over HTTPS (`tailscale serve --https=443 ...`) |
| `secureContext: true`, `navigator.gpu: MISSING` | Safari WebGPU flag off, or browser is Brave/Chrome on iOS | Enable the Safari flag; open in the real Safari app |
| `navigator.gpu present, adapter=null` | Low Power Mode / Lockdown Mode / GPU blocked | Turn both off |
| WebGL renderer names a GPU but WebGPU still missing | Normal: WebGL is not secure-context gated; only proves hardware | Not the issue; chase the rows above |
| Renders first frame, histogram shows, taps do nothing | Almost always one of the above (no `navigator.gpu`) | Run this page; fix the failing line |

## Touch events (separate from WebGPU)

Even with WebGPU working, touch interaction needs the interactive canvases to use
**Pointer Events** (`onPointerDown/Move/Up` + `setPointerCapture`), not
mouse-only handlers, or single-finger drag will not move the probe on a
touchscreen. The VI/DP/FFT canvases in `js/show4dstem/index.tsx` follow this;
mirror it for any new interactive canvas. Verify with Chrome touch emulation
(`Emulation.setTouchEmulationEnabled`, `Input.dispatchTouchEvent`), but remember
emulation is not iOS WebKit: confirm the real device too.
