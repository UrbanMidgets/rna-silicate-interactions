const state = {
  records: [],
  viewer: null,
  activeGroup: null,
  activeFile: null,
  preferTrajectory: false,
  showSurface: true,
};

const selects = {
  surface: document.getElementById("surface"),
  system: document.getElementById("system"),
  role: document.getElementById("role"),
  frame: document.getElementById("frame"),
  state: document.getElementById("state"),
};

const toggles = {
  trajectory: document.getElementById("prefer-trajectory"),
  surface: document.getElementById("show-surface"),
};

const matchCountEl = document.getElementById("match-count");
const groupCountEl = document.getElementById("group-count");
const selectionMeta = document.getElementById("selection-meta");
const fileLinks = document.getElementById("file-links");

function uniqueValues(records, key) {
  return [...new Set(records.map((r) => r[key]).filter(Boolean))].sort();
}

function setSelectOptions(selectEl, values) {
  const current = selectEl.value;
  selectEl.innerHTML = "";
  const allOpt = document.createElement("option");
  allOpt.value = "all";
  allOpt.textContent = "All";
  selectEl.appendChild(allOpt);
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    selectEl.appendChild(opt);
  }
  if (values.includes(current)) {
    selectEl.value = current;
  } else {
    selectEl.value = "all";
  }
}

function getFilters() {
  return {
    surface: selects.surface.value,
    system: selects.system.value,
    role: selects.role.value,
    frame: selects.frame.value,
    state: selects.state.value,
  };
}

function applyFilters(records, filters) {
  return records.filter((r) => {
    return (
      (filters.surface === "all" || r.surface === filters.surface) &&
      (filters.system === "all" || r.system === filters.system) &&
      (filters.role === "all" || r.role === filters.role) &&
      (filters.frame === "all" || r.frame === filters.frame) &&
      (filters.state === "all" || r.state === filters.state)
    );
  });
}

function groupRecords(records) {
  const map = new Map();
  for (const r of records) {
    const key = [r.surface, r.system, r.role, r.frame, r.state].join("||");
    if (!map.has(key)) {
      map.set(key, {
        key,
        surface: r.surface,
        system: r.system,
        role: r.role,
        frame: r.frame,
        state: r.state,
        files: [],
      });
    }
    map.get(key).files.push(r);
  }
  return [...map.values()].sort((a, b) => a.key.localeCompare(b.key));
}

function pickStructure(group) {
  const structures = group.files.filter((f) => f.file_type === "structure" || f.file_type === "trajectory");
  if (!structures.length) return null;

  if (state.preferTrajectory) {
    const trj = structures.find((f) => f.file_type === "trajectory");
    if (trj) return trj;
  } else {
    const static = structures.find((f) => f.file_type === "structure");
    if (static) return static;
  }

  // Fallback to best available
  const sysPrefix = (group.system || "").toLowerCase() + "_";
  return (
    structures.find((s) => s.file_name.toLowerCase().startsWith(sysPrefix)) ||
    structures.find((s) => s.file_name.toLowerCase() !== `${group.frame}.xyz`.toLowerCase()) ||
    structures[0]
  );
}

async function loadFileIntoViewer(file) {
  state.activeFile = file;
  const response = await fetch(file.web_path);
  if (!response.ok) {
    throw new Error(`Could not load file (${response.status}) from ${file.web_path}`);
  }
  const xyzText = await response.text();

  state.viewer.clear();
  const model = state.viewer.addModel(xyzText, "xyz");
  const atomCount = model && typeof model.selectedAtoms === "function" ? model.selectedAtoms({}).length : 0;

  if (!atomCount) {
    throw new Error(`Parsed 0 atoms from ${file.file_name}`);
  }

  // Nucleotide: Stick + Sphere
  state.viewer.setStyle({ elem: ["C", "N", "P"] }, { stick: { radius: 0.18 }, sphere: { scale: 0.25 } });
  
  // Surface: Smaller sticks, more transparent or muted if needed
  if (state.showSurface) {
    state.viewer.setStyle({ elem: ["Si", "Al", "O"] }, { stick: { radius: 0.12, opacity: 0.9 } });
  } else {
    state.viewer.setStyle({ elem: ["Si", "Al", "O"] }, {}); // Hide
  }

  // Hydrogens: Line
  state.viewer.setStyle({ elem: "H" }, { line: { linewidth: 0.5 } });
  
  state.viewer.zoomTo();
  state.viewer.render();

  updateSelectionPanel();
}

function updateSelectionPanel() {
  const group = state.activeGroup;
  if (!group) return;

  selectionMeta.textContent = `${group.surface} | ${group.system} | ${group.role} | ${group.frame} | ${group.state} | atoms: ${state.viewer.selectedAtoms({}).length}`;

  fileLinks.innerHTML = "";
  const sorted = [...group.files].sort((a, b) => a.file_name.localeCompare(b.file_name));
  
  for (const file of sorted) {
    const isRenderable = file.file_type === "structure" || file.file_type === "trajectory";
    
    const li = document.createElement("li");
    li.className = "file-item" + (state.activeFile === file ? " active" : "");
    
    const btn = document.createElement("button");
    btn.textContent = file.file_name;
    if (isRenderable) {
      btn.onclick = () => loadFileIntoViewer(file).catch(err => alert(err));
    } else {
      btn.disabled = true;
      btn.style.color = "var(--muted)";
    }
    
    const ext = document.createElement("span");
    ext.className = "file-ext";
    ext.textContent = file.file_type;
    
    const gh = document.createElement("a");
    gh.className = "github-link";
    gh.href = file.blob_url || file.web_path;
    gh.target = "_blank";
    gh.innerHTML = " &boxplus;";
    gh.title = "View on GitHub";

    li.appendChild(btn);
    li.appendChild(ext);
    li.appendChild(gh);
    fileLinks.appendChild(li);
  }
}

async function refresh() {
  const filters = getFilters();
  const filtered = applyFilters(state.records, filters);
  const groups = groupRecords(filtered);
  matchCountEl.textContent = String(filtered.length);
  groupCountEl.textContent = String(groups.length);

  if (!groups.length) {
    selectionMeta.textContent = "No matching groups for the current filters.";
    fileLinks.innerHTML = "";
    state.viewer.clear();
    state.viewer.render();
    state.activeGroup = null;
    state.activeFile = null;
    return;
  }

  state.activeGroup = groups[0];
  const structure = pickStructure(groups[0]);
  if (structure) {
    await loadFileIntoViewer(structure);
  } else {
    selectionMeta.textContent = "No displayable structure in group.";
    fileLinks.innerHTML = "";
    state.viewer.clear();
    state.viewer.render();
    state.activeFile = null;
    updateSelectionPanel();
  }
}

async function init() {
  const response = await fetch("./data_index.json");
  const payload = await response.json();
  state.records = payload.records || [];

  state.viewer = $3Dmol.createViewer("viewer", {
    backgroundColor: "#ffffff",
  });

  window.addEventListener("resize", () => {
    if (state.viewer) state.viewer.resize();
  });

  setSelectOptions(selects.surface, uniqueValues(state.records, "surface"));
  setSelectOptions(selects.system, uniqueValues(state.records, "system"));
  setSelectOptions(selects.role, uniqueValues(state.records, "role"));
  setSelectOptions(selects.frame, uniqueValues(state.records, "frame"));
  setSelectOptions(selects.state, uniqueValues(state.records, "state"));

  Object.values(selects).forEach((el) => {
    el.addEventListener("change", () => refresh().catch(console.error));
  });

  toggles.trajectory.addEventListener("change", (e) => {
    state.preferTrajectory = e.target.checked;
    refresh().catch(console.error);
  });

  toggles.surface.addEventListener("change", (e) => {
    state.showSurface = e.target.checked;
    if (state.activeFile) {
      loadFileIntoViewer(state.activeFile).catch(console.error);
    }
  });

  setTimeout(() => {
    refresh().catch(console.error);
  }, 100);
}

init().catch((err) => {
  selectionMeta.textContent = `Failed to initialize app: ${err}`;
});

init().catch((err) => {
  selectionMeta.textContent = `Failed to initialize app: ${err}`;
});
