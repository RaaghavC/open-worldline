import "./style.css";
import { mountIcons, icon } from "./icons.js";
import { WorldRenderer } from "./renderer.js";

mountIcons();
const $ = (s) => document.querySelector(s),
  $$ = (s) => [...document.querySelectorAll(s)];
let world = null,
  busy = false,
  brush = "",
  toastTimer,
  busyTimer,
  requestStart = 0;
const names = ["Alpine landscape", "Desert dunes", "Alien landscape"];
const descriptions = [
  "An alpine lake beneath snow-capped mountains",
  "A desert of sandstone canyons and quiet dunes",
  "An alien valley with bioluminescent plants and crystals",
];
const timeLabel = (n) =>
  n < 1000 ? `${Math.round(n)} ms` : `${(n / 1000).toFixed(2)} s`;
let explorer;

function toast(message, error = false) {
  clearTimeout(toastTimer);
  $("#toast").textContent = message;
  $("#toast").classList.toggle("error", error);
  $("#toast").hidden = false;
  toastTimer = setTimeout(
    () => ($("#toast").hidden = true),
    error ? 9000 : 4200,
  );
}
function setBusy(
  value,
  title = "Generating terrain",
  description = "The neural model is generating the landscape.",
  overlay = true,
) {
  busy = value;
  $$("button").forEach((b) => {
    if (
      !b.closest("dialog") &&
      b.id !== "collapseButton" &&
      b.id !== "hideUIButton"
    )
      b.disabled = value;
  });
  $("#busy").hidden = !value || !overlay;
  clearInterval(busyTimer);
  if (value) {
    requestStart = performance.now();
    $("#busyTitle").textContent = title;
    $("#busyText").textContent = description;
    $("#busyDuration").textContent = "0.0 s";
    busyTimer = setInterval(() => {
      $("#busyDuration").textContent =
        `${((performance.now() - requestStart) / 1000).toFixed(1)} s`;
    }, 100);
  }
}
async function api(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    let message = data.detail;
    if (Array.isArray(message)) message = message.map((e) => e.msg).join(". ");
    throw new Error(message || `The local app returned ${response.status}.`);
  }
  return data;
}
async function operation(
  fn,
  { title, description, overlay = true, success } = {},
) {
  if (busy) return;
  setBusy(true, title, description, overlay);
  try {
    await fn();
    if (success) toast(success);
  } catch (error) {
    toast(error.message, true);
    console.error(error);
  } finally {
    setBusy(false);
  }
}
async function applyWorld(next, resetCamera = false) {
  world = next;
  explorer?.setWorld(world.state, { resetCamera });
  localStorage.setItem("worldline-last-world", world.id);
  $("#worldTitle").textContent = world.parent
    ? world.name
    : names[world.state.biome];
  $("#worldSubtitle").textContent = world.parent
    ? "AN ALTERNATE FUTURE"
    : "AN ORIGINAL NEURAL LANDSCAPE";
  $("#worldSeed").textContent = `SEED ${world.state.seed}`;
  $("#worldRevision").textContent = `REVISION ${world.revision} · SAVED`;
  $("#tickValue").textContent = world.state.tick;
  for (const channel of ["rain", "heat"]) {
    $(`#${channel}`).value = world.state.weather[channel];
    $(`#${channel}Value`).textContent =
      `${Math.round(world.state.weather[channel] * 100)}%`;
  }
  $("#prompt").value = world.state.prompt;
  $("#seed").value = world.state.seed;
  $$("[data-biome]").forEach((b) =>
    b.classList.toggle("active", Number(b.dataset.biome) === world.state.biome),
  );
  if (next.generation_ms !== undefined)
    $("#genLatency").textContent = timeLabel(next.generation_ms);
  else if (resetCamera) $("#genLatency").textContent = "Not timed";
  await loadHistory();
}
function selectTab(name) {
  $$("[data-tab]").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === name);
    b.setAttribute("aria-selected", b.dataset.tab === name ? "true" : "false");
  });
  $$("[data-pane]").forEach((p) =>
    p.classList.toggle("active", p.dataset.pane === name),
  );
  if (name !== "shape") setBrush("");
  if (name === "history") loadHistory().catch((e) => toast(e.message, true));
}
function setBrush(kind) {
  brush = kind;
  $$("[data-brush]").forEach((b) =>
    b.classList.toggle("active", b.dataset.brush === kind),
  );
  document.body.classList.toggle("editing", !!kind);
  explorer?.setBrush(kind, Number($("#radius").value));
  if (kind)
    $("#controlsHint").textContent =
      `Click the landscape to ${kind} · Scroll to zoom`;
  else updateMode(explorer?.mode || "orbit");
}
function updateMode(mode) {
  $("#orbitButton").classList.toggle("active", mode === "orbit");
  $("#walkButton").classList.toggle("active", mode === "walk");
  $("#controlsHint").innerHTML =
    mode === "orbit"
      ? "Drag to orbit <span>·</span> Scroll to zoom"
      : "W A S D to move <span>·</span> Drag to look <span>·</span> Q / E down / up";
}
function requireWorld() {
  if (world) return true;
  toast("Generate or import a world first.");
  return false;
}

try {
  explorer = new WorldRenderer($("#scene"), {
    onFps: (n) => ($("#renderFps").textContent = n),
    onDirection: (a) =>
      ($("#compassNeedle").style.transform = `rotate(${a}rad)`),
    onMode: updateMode,
    onPick: ({ x, z }) => {
      if (!world || busy || !brush) return;
      const kind = brush;
      operation(
        async () => {
          const next = await api(`/api/worlds/${world.id}/edit`, {
            kind,
            x,
            z,
            radius: Number($("#radius").value),
            strength: Number($("#strength").value),
            revision: world.revision,
          });
          await applyWorld(next);
        },
        {
          overlay: false,
          success: `${kind[0].toUpperCase() + kind.slice(1)} applied. Revision saved.`,
        },
      );
    },
  });
} catch (error) {
  toast(
    `3D graphics could not start: ${error.message}. Try a browser with WebGL enabled.`,
    true,
  );
  console.error(error);
}

$$("[data-tab]").forEach((b) =>
  b.addEventListener("click", () => selectTab(b.dataset.tab)),
);
$$("[data-biome]").forEach((b) =>
  b.addEventListener("click", () => {
    $$("[data-biome]").forEach((p) => p.classList.toggle("active", p === b));
    $("#prompt").value = descriptions[Number(b.dataset.biome)];
  }),
);
$$("[data-brush]").forEach((b) =>
  b.addEventListener("click", () => setBrush(b.dataset.brush)),
);
$("#randomSeed").addEventListener("click", () => {
  $("#seed").value = crypto.getRandomValues(new Uint32Array(1))[0] % 2147483648;
});
$("#generateButton").addEventListener("click", () =>
  operation(
    async () => {
      const seed = Number($("#seed").value),
        prompt = $("#prompt").value.trim();
      if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647)
        throw new Error("Choose a whole-number seed from 0 to 2147483647.");
      if (!prompt)
        throw new Error("Enter a biome or choose Alpine, Desert or Alien.");
      const next = await api("/api/worlds", { prompt, seed, steps: 24 });
      await applyWorld(next, true);
    },
    {
      title: "Generating terrain",
      description: "Generating spatial fields with your local neural model.",
    },
  ),
);
$("#radius").addEventListener("input", () => {
  const r = Number($("#radius").value);
  $("#radiusValue").textContent =
    r < 0.06 ? "Small" : r < 0.13 ? "Medium" : "Large";
  explorer?.setBrush(brush, r);
});
for (const name of ["strength", "rain", "heat"])
  $(`#${name}`).addEventListener("input", () => {
    $(`#${name}Value`).textContent =
      `${Math.round(Number($(`#${name}`).value) * 100)}%`;
  });
$("#stepButton").addEventListener("click", () => {
  if (!requireWorld()) return;
  operation(
    async () => {
      const next = await api(`/api/worlds/${world.id}/step`, {
        rain: Number($("#rain").value),
        heat: Number($("#heat").value),
        steps: 10,
        revision: world.revision,
      });
      await applyWorld(next);
      toast(
        `10 steps predicted in ${timeLabel(next.simulation_ms)}. Saved as revision ${next.revision}.`,
      );
    },
    {
      title: "Predicting ecology changes",
      description:
        "Predicting water, growth and heat over 10 simulation steps.",
    },
  );
});
$("#branchButton").addEventListener("click", () => {
  if (!requireWorld()) return;
  operation(
    async () => {
      const next = await api(`/api/worlds/${world.id}/branch`, {
        name: `${names[world.state.biome]} · Future ${world.revision + 1}`,
        revision: world.revision,
      });
      await applyWorld(next);
    },
    {
      title: "Creating a world branch",
      description: "Saving an exact copy of this moment.",
      success: "Alternate future created. The original world is in My worlds.",
    },
  );
});
$("#refreshHistory").addEventListener("click", () =>
  loadHistory().catch((e) => toast(e.message, true)),
);

function eventTitle(event, revision) {
  if (event.type === "generate") return "World generated";
  if (event.type === "edit")
    return `${event.kind[0].toUpperCase() + event.kind.slice(1)} brush`;
  if (event.type === "simulate") return `${event.steps} simulation steps`;
  if (event.type === "branch")
    return `Branched from revision ${event.revision}`;
  if (event.type === "restore") return `Restored revision ${event.revision}`;
  if (event.type === "import") return "World imported";
  return `Revision ${revision}`;
}
async function loadHistory() {
  if (!world) return;
  const id = world.id,
    items = await api(`/api/worlds/${id}/history`);
  if (world.id !== id) return;
  const list = $("#historyList");
  list.replaceChildren();
  for (const item of items) {
    const row = document.createElement("div");
    row.className = "history-item";
    const node = document.createElement("span");
    node.className = "history-node";
    const body = document.createElement("div");
    const title = document.createElement("span");
    title.className = "history-name";
    title.textContent = eventTitle(item.event, item.revision);
    const sub = document.createElement("span");
    sub.className = "history-date";
    sub.textContent = `Revision ${item.revision} · ${new Date(item.created * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
    body.append(title, sub);
    row.append(node, body);
    if (item.revision !== world.revision) {
      const button = document.createElement("button");
      button.textContent = "Restore";
      button.setAttribute("aria-label", `Restore revision ${item.revision}`);
      button.addEventListener("click", () =>
        operation(
          async () => {
            await applyWorld(
              await api(`/api/worlds/${world.id}/restore`, {
                target: item.revision,
                revision: world.revision,
              }),
            );
          },
          {
            title: "Returning to a saved moment",
            description: "Restoring the exact terrain and ecology.",
            success: `Revision ${item.revision} restored. Later moments remain in history.`,
          },
        ),
      );
      row.append(button);
    } else {
      const label = document.createElement("small");
      label.textContent = "Now";
      label.className = "history-date";
      row.append(label);
    }
    list.append(row);
  }
}

async function openLibrary() {
  const dialog = $("#libraryDialog");
  dialog.showModal();
  const list = $("#worldList");
  list.textContent = "Loading saved worlds…";
  try {
    const items = await api("/api/worlds");
    list.replaceChildren();
    if (!items.length) {
      const p = document.createElement("p");
      p.className = "empty";
      p.textContent = "No saved worlds yet. Generate or import a world.";
      list.append(p);
    }
    for (const item of items) {
      const button = document.createElement("button");
      button.className = "world-card";
      const mark = document.createElement("span");
      mark.className = "world-icon";
      mark.innerHTML = icon(item.parent ? "branch" : "mountain");
      const details = document.createElement("span");
      details.className = "details";
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = item.name;
      const date = document.createElement("small");
      date.textContent = `${new Date(item.created * 1000).toLocaleDateString()} · Revision ${item.revision}${item.parent ? " · Alternate future" : ""}`;
      details.append(name, date);
      button.append(mark, details);
      if (item.id === world?.id) {
        const current = document.createElement("span");
        current.className = "current-label";
        current.textContent = "Open";
        button.append(current);
      }
      button.addEventListener("click", () => {
        dialog.close();
        operation(
          async () => {
            await applyWorld(await api(`/api/worlds/${item.id}`), true);
          },
          {
            title: "Loading your saved world",
            description: "Loading its saved terrain, ecology and history.",
          },
        );
      });
      list.append(button);
    }
  } catch (error) {
    list.textContent = error.message;
  }
}
$("#libraryButton").addEventListener("click", openLibrary);
$("#aboutButton").addEventListener("click", () =>
  $("#aboutDialog").showModal(),
);
$$(".close-dialog").forEach((b) =>
  b.addEventListener("click", () => b.closest("dialog").close()),
);
$$("dialog").forEach((d) =>
  d.addEventListener("click", (e) => {
    if (e.target === d) {
      const r = d.getBoundingClientRect();
      if (
        e.clientX < r.left ||
        e.clientX > r.right ||
        e.clientY < r.top ||
        e.clientY > r.bottom
      )
        d.close();
    }
  }),
);
function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("#exportButton").addEventListener("click", () => {
  if (!requireWorld()) return;
  operation(
    async () => {
      const data = await api(`/api/worlds/${world.id}/export`);
      download(
        new Blob([JSON.stringify(data)], { type: "application/json" }),
        `worldline-${world.state.seed}-r${world.revision}.json`,
      );
    },
    {
      overlay: false,
      success:
        "World exported with terrain, ecology, seed and simulation state.",
    },
  );
});
$("#importButton").addEventListener("click", () => $("#importFile").click());
$("#importFile").addEventListener("change", () => {
  const file = $("#importFile").files?.[0];
  if (!file) return;
  operation(
    async () => {
      if (file.size > 2000000)
        throw new Error("Choose a Worldline JSON file smaller than 2 MB.");
      let data;
      try {
        data = JSON.parse(await file.text());
      } catch {
        throw new Error(
          "This file is not valid JSON. Choose a Worldline world export.",
        );
      }
      if (!data.state)
        throw new Error(
          "This file is missing its world state. Choose a Worldline world export.",
        );
      await applyWorld(
        await api("/api/import", {
          state: data.state,
          name: data.name || "Imported world",
        }),
        true,
      );
      $("#importFile").value = "";
    },
    {
      title: "Importing your world",
      description: "Checking and loading the exported world.",
      success: "World imported and saved locally.",
    },
  );
});
$("#captureButton").addEventListener("click", async () => {
  if (!explorer || !requireWorld()) return;
  try {
    const blob = await explorer.screenshot();
    if (!blob) throw new Error("The screenshot could not be saved.");
    download(blob, `worldline-${world.state.seed}.png`);
    toast("Landscape screenshot saved.");
  } catch (error) {
    toast(error.message, true);
  }
});
$("#orbitButton").addEventListener("click", () => {
  setBrush("");
  explorer?.setMode("orbit");
  updateMode("orbit");
});
$("#walkButton").addEventListener("click", () => {
  setBrush("");
  explorer?.setMode("walk");
  updateMode("walk");
});
$("#homeViewButton").addEventListener("click", () => {
  setBrush("");
  explorer?.home();
  updateMode("orbit");
});
$("#hideUIButton").addEventListener("click", () => {
  document.body.classList.toggle("hide-ui");
  if (document.body.classList.contains("hide-ui"))
    toast("Press H to show the interface.");
});
addEventListener("keydown", (e) => {
  if (
    e.code === "KeyH" &&
    !["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName) &&
    !document.querySelector("dialog[open]")
  )
    document.body.classList.toggle("hide-ui");
});
$("#collapseButton").addEventListener("click", () => {
  const collapsed = $(".studio").classList.toggle("collapsed");
  $("#collapseButton").setAttribute(
    "aria-label",
    collapsed ? "Expand controls" : "Collapse controls",
  );
  $("#collapseButton").innerHTML = icon(collapsed ? "spark" : "minus");
});

async function init() {
  try {
    const [status, worlds] = await Promise.all([
      api("/api/status"),
      api("/api/worlds"),
    ]);
    $("#deviceName").textContent =
      status.device === "mps"
        ? "Apple Silicon"
        : status.device === "cuda"
          ? "NVIDIA GPU"
          : "CPU";
    $("#modelStatus").textContent = status.trained
      ? "Original model · runs locally"
      : "Model needs training";
    if (worlds.length) {
      const stored = localStorage.getItem("worldline-last-world");
      const found = worlds.find((w) => w.id === stored) || worlds[0];
      await operation(
        async () => {
          await applyWorld(await api(`/api/worlds/${found.id}`), true);
        },
        {
          title: "Welcome back to your world",
          description: "Loading its saved landscape and history.",
        },
      );
    } else if (status.trained) {
      await operation(async () => {
        await applyWorld(
          await api("/api/worlds", {
            prompt: descriptions[0],
            seed: 42,
            steps: 24,
          }),
          true,
        );
      });
    } else {
      toast(
        "Trained weights are missing. Run python -m worldline.train, then reload this page.",
        true,
      );
      $("#worldTitle").textContent = "Model needs training";
    }
  } catch (error) {
    $("#modelStatus").textContent = "Local server unavailable";
    toast(
      `Cannot connect to Worldline. Start its local server and reload. ${error.message}`,
      true,
    );
  }
}
init();
