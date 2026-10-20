export async function loadPose() {
  const errs = [];
  for (const v of MEDIAPIPE_VERSIONS) {
    const base = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${v}`;
    let mod, fs;
    try { mod = await import(base); fs = await mod.FilesetResolver.forVisionTasks(base + "/wasm"); }
    catch (e) { errs.push(`${v} load: ${e.message || e}`); continue; }
    for (const delegate of ["GPU", "CPU"]) {
      try {
        lm = await mod.PoseLandmarker.createFromOptions(fs, { baseOptions: { modelAssetPath: PROFILE.pose, delegate }, runningMode: "VIDEO", numPoses: 1 });
        return `MediaPipe ${v} (${delegate})`;
      } catch (e) { errs.push(`${v} ${delegate}: ${e.message || e}`); }
    }
  }
  throw new Error(errs.join(" | "));
}

export function detect(video, t) {
  const r = lm.detectForVideo(video, t);
  if (!r.landmarks.length) return null;
  const f = (L, F) => L.map((p, i) => ({ x: F[i][0].f(p.x, t), y: F[i][1].f(p.y, t), z: F[i][2] ? F[i][2].f(p.z, t) : 0, visibility: p.visibility }));
  return { img: f(r.landmarks[0], FI), world: f(r.worldLandmarks[0], FW) };
}
