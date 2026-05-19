const state = {
  records: [],
  viewer: null,
  activeGroup: null,
  activeFile: null,
  preferTrajectory: false,
  showSurface: true,
};

// Elements will be populated during init.
let els = {};

function uniqueValues(records, key) {
  return [...new Set(records.map((r) => r[key]).filter(Boolean))].sort();
}

function setSelectOptions(selectEl, values) {
  if (!selectEl) {
    console.warn("setSelectOptions: target element missing");
    return;
  }
  const current = selectEl.value;
  selectEl.innerHTML = "";
  
  const allOpt = document.createElement("option");
  allOpt.value = "all";
  allOpt.textContent = "All";
  selectEl.appendChild(allOpt);
  
  if (Array.isArray(values)) {
    for (const v of values) {
      if (!v) continue;
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      selectEl.appendChild(opt);
    }
  }
  
  if (values && values.includes(current)) {
    selectEl.value = current;
  } else {
    selectEl.value = "all";
  }
}

function getFilters() {
  return {
    surface: els.surface.value,
    system: els.system.value,
    role: els.role.value,
    frame: els.frame.value,
    state: els.state.value,
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
  const sys = (group.system || "").toLowerCase();
  const baseSys = sys.split("_")[0];
  
  const byFullSystem = structures.find((s) => s.file_name.toLowerCase().startsWith(sys + "_"));
  if (byFullSystem) return byFullSystem;

  const byBaseSystem = structures.find((s) => s.file_name.toLowerCase().startsWith(baseSys));
  if (byBaseSystem) return byBaseSystem;

  return (
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

  els.selectionMeta.textContent = `${group.surface} | ${group.system} | ${group.role} | ${group.frame} | ${group.state} | atoms: ${state.viewer.selectedAtoms({}).length}`;

  els.fileLinks.innerHTML = "";
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
    els.fileLinks.appendChild(li);
  }
}

async function refresh() {
  const filters = getFilters();
  const filtered = applyFilters(state.records, filters);
  const groups = groupRecords(filtered);
  els.matchCount.textContent = String(filtered.length);
  els.groupCount.textContent = String(groups.length);

  if (!groups.length) {
    els.selectionMeta.textContent = "No matching groups for the current filters.";
    els.fileLinks.innerHTML = "";
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
    els.selectionMeta.textContent = "No displayable structure in group.";
    els.fileLinks.innerHTML = "";
    state.viewer.clear();
    state.viewer.render();
    state.activeFile = null;
    updateSelectionPanel();
  }
}

async function init() {
  console.log("Initializing app...");
  // Populate elements map
  els = {
    surface: document.getElementById("surface"),
    system: document.getElementById("system"),
    role: document.getElementById("role"),
    frame: document.getElementById("frame"),
    state: document.getElementById("state"),
    trajectory: document.getElementById("prefer-trajectory"),
    showSurfaceToggle: document.getElementById("show-surface"),
    matchCount: document.getElementById("match-count"),
    groupCount: document.getElementById("group-count"),
    selectionMeta: document.getElementById("selection-meta"),
    fileLinks: document.getElementById("file-links"),
  };

  if (els.selectionMeta) {
    els.selectionMeta.textContent = "Loading data index...";
  }

  try {
    const response = await fetch(`./data_index.json?v=${Date.now()}`);
    if (!response.ok) throw new Error(`HTTP ${response.status} loading index`);
    const payload = await response.json();
    state.records = payload.records || [];
    console.log(`Loaded ${state.records.length} records.`);

    if (!state.records.length) {
      els.selectionMeta.textContent = "Warning: Loaded 0 records from index.";
      return;
    }
  } catch (err) {
    console.error("Failed to fetch index:", err);
    if (els.selectionMeta) els.selectionMeta.textContent = `Error loading index: ${err.message}`;
    return;
  }

  try {
    state.viewer = $3Dmol.createViewer("viewer", {
      backgroundColor: "#ffffff",
    });
    console.log("3Dmol viewer created.");
  } catch (err) {
    console.error("Failed to create 3Dmol viewer:", err);
    if (els.selectionMeta) els.selectionMeta.textContent = `Error creating 3D viewer: ${err.message}`;
    // Continue anyway to see if dropdowns populate
  }

  window.addEventListener("resize", () => {
    if (state.viewer) state.viewer.resize();
  });

  const surfaces = uniqueValues(state.records, "surface");
  const systems = uniqueValues(state.records, "system");
  console.log("Available surfaces:", surfaces);
  console.log("Available systems:", systems);

  setSelectOptions(els.surface, surfaces);
  setSelectOptions(els.system, systems);
  setSelectOptions(els.role, uniqueValues(state.records, "role"));
  setSelectOptions(els.frame, uniqueValues(state.records, "frame"));
  setSelectOptions(els.state, uniqueValues(state.records, "state"));

  const filterEls = [els.surface, els.system, els.role, els.frame, els.state];
  filterEls.forEach((el) => {
    if (el) el.addEventListener("change", () => refresh().catch(console.error));
  });

  if (els.trajectory) {
    els.trajectory.addEventListener("change", (e) => {
      state.preferTrajectory = e.target.checked;
      refresh().catch(console.error);
    });
  }

  if (els.showSurfaceToggle) {
    els.showSurfaceToggle.addEventListener("change", (e) => {
      state.showSurface = e.target.checked;
      if (state.activeFile) {
        loadFileIntoViewer(state.activeFile).catch(console.error);
      }
    });
  }

  setTimeout(() => {
    refresh().catch(err => {
      console.error("Initial refresh failed:", err);
      if (els.selectionMeta) els.selectionMeta.textContent = `Error: ${err.message}`;
    });
  }, 100);
}

init().catch((err) => {
  console.error("App initialization failed:", err);
  if (els.selectionMeta) {
    els.selectionMeta.textContent = `Failed to initialize app: ${err.message}`;
  }
});
