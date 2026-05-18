const state = {
  records: [],
  viewer: null,
};

const selects = {
  surface: document.getElementById("surface"),
  system: document.getElementById("system"),
  role: document.getElementById("role"),
  frame: document.getElementById("frame"),
  state: document.getElementById("state"),
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
  const structures = group.files.filter((f) => f.file_type === "structure");
  if (!structures.length) return null;

  // Prefer explicit system-named structures (amp_/cmp_/gmp_/ump_) over generic frameXX.xyz.
  const sysPrefix = (group.system || "").toLowerCase() + "_";
  const bySystemName = structures.find((s) => s.file_name.toLowerCase().startsWith(sysPrefix));
  if (bySystemName) return bySystemName;

  // Next prefer non-generic names to avoid selecting placeholder frame snapshots.
  const nonGeneric = structures.find((s) => s.file_name.toLowerCase() !== `${group.frame}.xyz`.toLowerCase());
  if (nonGeneric) return nonGeneric;

  const generic = structures.find((s) => s.file_name.toLowerCase() === `${group.frame}.xyz`.toLowerCase());
  return generic || structures[0];
}

async function renderGroup(group) {
  const structure = pickStructure(group);
  if (!structure) {
    selectionMeta.textContent = "No structure file in selected group.";
    fileLinks.innerHTML = "";
    state.viewer.clear();
    state.viewer.render();
    return;
  }

  const response = await fetch(structure.web_path);
  if (!response.ok) {
    throw new Error(`Could not load structure (${response.status}) from ${structure.web_path}`);
  }
  const xyzText = await response.text();

  state.viewer.clear();
  const model = state.viewer.addModel(xyzText, "xyz");
  const atomCount = model && typeof model.selectedAtoms === "function" ? model.selectedAtoms({}).length : 0;

  if (!atomCount) {
    throw new Error(`Parsed 0 atoms from ${structure.file_name}`);
  }

  // Primary style.
  state.viewer.setStyle({}, { stick: { radius: 0.18 }, sphere: { scale: 0.25 } });
  // Fallback line style so very large systems still remain visible on weaker GPUs.
  state.viewer.setStyle({ elem: "H" }, { line: { linewidth: 0.5 } });
  state.viewer.zoomTo();
  state.viewer.render();

  selectionMeta.textContent = `${group.surface} | ${group.system} | ${group.role} | ${group.frame} | ${group.state} | atoms: ${atomCount} | ${structure.file_name}`;

  fileLinks.innerHTML = "";
  const sorted = [...group.files].sort((a, b) => a.file_name.localeCompare(b.file_name));
  for (const file of sorted) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = file.blob_url || file.web_path;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.textContent = `${file.file_name} (${file.file_type})`;
    li.appendChild(a);
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
    return;
  }

  await renderGroup(groups[0]);
}

async function init() {
  const response = await fetch("./data_index.json");
  const payload = await response.json();
  state.records = payload.records || [];

  state.viewer = $3Dmol.createViewer("viewer", {
    backgroundColor: "#f7fafc",
  });

  setSelectOptions(selects.surface, uniqueValues(state.records, "surface"));
  setSelectOptions(selects.system, uniqueValues(state.records, "system"));
  setSelectOptions(selects.role, uniqueValues(state.records, "role"));
  setSelectOptions(selects.frame, uniqueValues(state.records, "frame"));
  setSelectOptions(selects.state, uniqueValues(state.records, "state"));

  Object.values(selects).forEach((el) => {
    el.addEventListener("change", () => {
      refresh().catch((err) => {
        selectionMeta.textContent = `Error: ${err}`;
      });
    });
  });

  await refresh();
}

init().catch((err) => {
  selectionMeta.textContent = `Failed to initialize app: ${err}`;
});
