const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const context = canvas.getContext("2d");
let room = null;
let busy = false;
let saved = null;
let held = null;
const keyActions = {
  w: 1,
  ArrowUp: 1,
  s: 2,
  ArrowDown: 2,
  a: 3,
  ArrowLeft: 3,
  d: 4,
  ArrowRight: 4,
  e: 5,
  " ": 0,
};
const names = [
  "Wait",
  "Forward",
  "Backward",
  "Turn left",
  "Turn right",
  "Open / close",
];

function controls() {
  document.querySelectorAll("[data-action]").forEach((button) => {
    button.disabled = !room || busy;
  });
  $("branch").disabled = !room || busy;
  $("return").disabled = !saved || busy;
  $("start").disabled = busy;
}
async function request(path, body) {
  let response;
  try {
    response = await fetch(
      path,
      body === undefined
        ? {}
        : {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          },
    );
  } catch {
    throw new Error(
      "Room Lab is not responding. Check that its local server is running, then try again.",
    );
  }
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(
      "Room Lab returned an unreadable response. Check its local server, then try again.",
    );
  }
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "Check the seed and try again.",
    );
  return data;
}
async function show(data) {
  const image = new Image();
  image.src = data.image;
  await image.decode();
  context.drawImage(image, 0, 0);
  room = data;
  $("placeholder").hidden = true;
  $("source").textContent = data.revision
    ? "Neural prediction"
    : "Initial rendered view";
  $("frame").textContent = `Frame ${data.revision}`;
  $("timing").textContent =
    data.inference_ms === null
      ? "Every action generates one image"
      : `${data.inference_ms.toFixed(0)} ms · model and history update`;
}
async function perform(operation) {
  if (busy) return;
  busy = true;
  controls();
  try {
    await operation();
  } catch (error) {
    held = null;
    $("status").textContent = error.message;
  } finally {
    busy = false;
    controls();
  }
}
$("setup").addEventListener("submit", (event) => {
  event.preventDefault();
  held = null;
  perform(async () => {
    $("status").textContent =
      "Loading the original model and preparing the first view…";
    await show(
      await request("/api/rooms", {
        seed: Number($("seed").value),
        kind: $("model").value,
      }),
    );
    saved = null;
    $("return").hidden = true;
    $("start").textContent = "Restart room";
    $("status").textContent =
      "Room ready. Try opening the door, then turning left or right.";
    canvas.focus();
  });
});
async function step(action) {
  if (!room || busy) return;
  await perform(async () => {
    await show(
      await request(`/api/rooms/${room.id}/step`, {
        action,
        revision: room.revision,
      }),
    );
    $("status").textContent =
      `${names[action]} · ${room.revision} generated ${room.revision === 1 ? "image" : "images"}. ${room.revision >= 24 ? "Long rollouts often lose room geometry. Restart to compare." : ""}`;
  });
}
document.querySelectorAll("[data-action]").forEach((button) =>
  button.addEventListener("click", () => {
    held = null;
    canvas.focus();
    step(Number(button.dataset.action));
  }),
);
$("branch").addEventListener("click", () => {
  held = null;
  perform(async () => {
    saved = (
      await request(`/api/rooms/${room.id}/branch`, { revision: room.revision })
    ).id;
    $("return").hidden = false;
    $("status").textContent =
      `Saved frame ${room.revision}. Explore, then return to try different controls.`;
    canvas.focus();
  });
});
$("return").addEventListener("click", () => {
  held = null;
  perform(async () => {
    const checkpoint = await request(`/api/rooms/${saved}`);
    await show(
      await request(`/api/rooms/${saved}/branch`, {
        revision: checkpoint.revision,
      }),
    );
    $("status").textContent =
      `Returned to frame ${room.revision}. The saved branch is still available.`;
    canvas.focus();
  });
});
document.addEventListener("keydown", (event) => {
  if (
    ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(event.target.tagName) ||
    event.altKey ||
    event.ctrlKey ||
    event.metaKey
  )
    return;
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  if (!(key in keyActions) || !room) return;
  event.preventDefault();
  if (!event.repeat) {
    held = key;
    step(keyActions[key]);
  }
});
document.addEventListener("keyup", (event) => {
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  if (held === key) held = null;
});
window.addEventListener("blur", () => {
  held = null;
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) held = null;
});
document.addEventListener("focusin", (event) => {
  if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(event.target.tagName))
    held = null;
});
setInterval(() => {
  if (held !== null && !busy && !["e", " "].includes(held))
    step(keyActions[held]);
}, 160);
controls();
