// Облачный контроль — работа страницы.
// Здесь нет нейросети: страница только отправляет снимки на сервер (app.py) и показывает ответ.
"use strict";

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;   // textContent: имя файла не станет кодом страницы
  return node;
}

async function api(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (e) {
    throw new Error("Сервер не отвечает. Проверьте, что программа app.py запущена.");
  }
  if (!response.ok) {
    let detail = "Ошибка сервера (" + response.status + ")";
    try {
      const data = await response.json();
      if (typeof data.detail === "string") detail = data.detail;
    } catch (e) { /* ответ не в формате JSON */ }
    throw new Error(detail);
  }
  return response.json();
}

function percentText(value) {
  return value.toFixed(1).replace(".", ",") + " %";
}

function imagesWord(count) {
  const d = count % 10;
  const dd = count % 100;
  if (d === 1 && dd !== 11) return "снимок";
  if (d >= 2 && d <= 4 && (dd < 10 || dd >= 20)) return "снимка";
  return "снимков";
}

// ============================================================ переключение тем
const themeBtn = $("theme-toggle");

function applyTheme(isDark) {
  document.body.classList.toggle("light-theme", !isDark);
  if (themeBtn) {
    const icon = $("theme-icon");
    if (icon) icon.textContent = isDark ? "☾" : "☀";
    themeBtn.setAttribute("aria-pressed", String(isDark));
    themeBtn.setAttribute("aria-label", isDark ? "Включить светлую тему" : "Включить тёмную тему");
  }
  localStorage.setItem("theme", isDark ? "dark" : "light");
}

if (themeBtn) {
  const savedTheme = localStorage.getItem("theme");
  applyTheme(savedTheme ? savedTheme === "dark" : true);   // по умолчанию — «космическая» тёмная

  themeBtn.addEventListener("click", () => {
    applyTheme(document.body.classList.contains("light-theme"));
  });
}

// ============================================================ вкладки
const tabs = { process: $("tab-process"), label: $("tab-label") };
const panels = { process: $("panel-process"), label: $("panel-label") };

function showTab(name) {
  Object.keys(tabs).forEach((key) => {
    const active = key === name;
    tabs[key].classList.toggle("is-active", active);
    tabs[key].setAttribute("aria-selected", String(active));
    panels[key].hidden = !active;
  });
  if (name === "label") loadImageList();
}
tabs.process.addEventListener("click", () => showTab("process"));
tabs.label.addEventListener("click", () => showTab("label"));

// ============================================================ состояние модели и папки
async function refreshStatus() {
  const pill = $("model-pill");
  const pillText = $("model-pill-text");
  try {
    const s = await api("/api/status");
    const note = $("model-note");
    if (!s.model_ready) {
      note.textContent = "Модель не найдена. Обучите сеть: python train.py";
      note.classList.add("is-bad");
      if (pill) pill.classList.add("is-bad");
      if (pillText) pillText.textContent = "Модель не найдена";
    } else {
      const m = s.model;
      const when = m.trained_at ? m.trained_at.replace("T", " ").slice(0, 16) : "—";
      const real = m.real_pixels ? "дообучена на настоящих снимках (" + m.real_images.length + " шт.)" : "базовая (только учебные снимки)";
      note.textContent = "Нейросеть: " + real + ", обучена " + when;
      note.classList.remove("is-bad");
      if (pill) pill.classList.remove("is-bad");
      if (pillText) {
        const accuracy = Number(m.pixel_accuracy_holdout);
        pillText.textContent = "Нейросеть активна" + (accuracy
          ? " (" + (accuracy * 100).toFixed(1).replace(".", ",") + " % точности)"
          : "");
      }
    }
    const waiting = s.incoming_waiting.length;
    $("folder-box").hidden = waiting === 0;
    if (waiting) {
      $("folder-text").textContent = "В папке data/incoming уже лежат снимки: " + waiting + " шт. Их можно обработать без загрузки.";
    }
  } catch (e) {
    $("model-note").textContent = e.message;
    $("model-note").classList.add("is-bad");
    if (pill) pill.classList.add("is-bad");
    if (pillText) pillText.textContent = "Сервер не отвечает";
  }
}

// ============================================================ обработка снимков
let picked = [];

function setPicked(fileList) {
  picked = Array.from(fileList);
  const box = $("picked");
  if (!picked.length) {
    box.textContent = "Файлы не выбраны";
  } else {
    const names = picked.slice(0, 3).map((f) => f.name).join(", ");
    box.textContent = "Выбрано: " + picked.length + " — " + names + (picked.length > 3 ? " и ещё " + (picked.length - 3) : "");
  }
  $("drop").classList.toggle("has-files", picked.length > 0);
  $("run").disabled = picked.length === 0;
}

$("files").addEventListener("change", (e) => setPicked(e.target.files));
const drop = $("drop");
["dragenter", "dragover"].forEach((type) => drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
["dragleave", "drop"].forEach((type) => drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.remove("is-over"); }));
drop.addEventListener("drop", (e) => setPicked(e.dataTransfer.files));

// Порог: число, дуга на шкале и подпись под ней меняются вместе.
function updateThreshold(value) {
  const percent = Number(value);
  $("threshold-value").textContent = percent + " %";
  const arc = $("gauge-arc");
  if (arc) arc.style.strokeDasharray = Math.max(0.001, percent * 0.75) + " 100";
  const caption = $("threshold-caption");
  if (caption) {
    caption.textContent = percent === 0
      ? "Удалять все снимки с облаками"
      : "Удалить, если облачность > " + percent + " %";
  }
}

$("threshold").addEventListener("input", (e) => updateThreshold(e.target.value));
updateThreshold($("threshold").value);

function deleteMode() {
  return document.querySelector('input[name="delete-mode"]:checked').value;
}

function setStatus(text, isError) {
  const box = $("status");
  box.textContent = text;
  box.classList.toggle("is-error", Boolean(isError));
}

function setBusy(busy) {
  const runBtn = $("run");
  const spinner = $("run-spinner");
  const text = $("run-text");

  runBtn.disabled = busy || picked.length === 0;
  $("run-folder").disabled = busy;

  if (spinner && text) {
    spinner.hidden = !busy;
    text.textContent = busy ? "Обрабатываю…" : "Обработать снимки";
  } else {
    runBtn.textContent = busy ? "Обрабатываю…" : "Обработать снимки";
  }
}

function summaryCell(value, caption, className) {
  const cell = el("div", className);
  cell.appendChild(el("b", "", String(value)));
  cell.appendChild(el("span", "", caption));
  return cell;
}

function renderReport(data) {
  $("empty").hidden = true;
  $("report").hidden = false;

  const summary = $("summary");
  summary.replaceChildren(
    summaryCell(data.counts.total, "обработано"),
    summaryCell(data.counts.saved, "сохранено", "is-saved"),
    summaryCell(data.counts.deleted, "удалено", "is-deleted"),
    summaryCell(data.counts.errors, "не прочитано")
  );
  $("message").textContent = data.message;
  $("download-txt").href = "/api/report/" + data.run_id + "?format=txt";
  $("download-csv").href = "/api/report/" + data.run_id + "?format=csv";

  const rows = $("rows");
  rows.replaceChildren();
  data.results.forEach((r) => {
    const row = el("li", "row");

    const thumbs = el("div", "thumbs");
    if (r.preview) {
      const photo = el("img"); photo.src = "/previews/" + data.run_id + "/" + r.preview; photo.alt = "Снимок " + r.name; photo.loading = "lazy";
      const clouds = el("img"); clouds.src = "/previews/" + data.run_id + "/" + r.overlay; clouds.alt = "Облака на снимке закрашены"; clouds.loading = "lazy";
      thumbs.append(photo, clouds);
    }
    row.appendChild(thumbs);

    const middle = el("div");
    middle.appendChild(el("div", "row-name", r.name));
    if (r.status === "error") {
      middle.appendChild(el("div", "row-note", r.error));
    } else {
      const meter = el("div", "meter");
      meter.setAttribute("role", "img");
      meter.setAttribute("aria-label", "Облачность " + percentText(r.cloud_percent) + ", порог " + data.threshold + " %");
      const fill = el("div", "meter-fill"); fill.style.width = Math.max(0, Math.min(100, r.cloud_percent)) + "%";
      const mark = el("div", "meter-mark"); mark.style.left = data.threshold + "%";
      meter.append(fill, mark);
      middle.appendChild(meter);
      const text = el("div", "meter-text");
      const left = el("span"); left.append("облачность ", el("b", "", percentText(r.cloud_percent)));
      text.append(left, el("span", "", "порог " + data.threshold + " %"));
      middle.appendChild(text);
      if (r.saved_as && r.saved_as !== r.name) middle.appendChild(el("div", "row-note", "Сохранён под именем " + r.saved_as));
    }
    row.appendChild(middle);

    const labels = { saved: "Сохранён", deleted: r.moved_to_trash ? "В корзине" : "Удалён", error: "Не прочитан" };
    row.appendChild(el("div", "verdict " + r.status, labels[r.status]));
    rows.appendChild(row);
  });
}

async function runProcessing(request) {
  setBusy(true);
  setStatus("Нейросеть смотрит снимки…");
  try {
    const data = await request();
    renderReport(data);
    setStatus(data.counts.total ? "Готово за " + data.seconds.toFixed(1).replace(".", ",") + " с." : "В папке не нашлось снимков.");
    $("files").value = "";
    setPicked([]);
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    setBusy(false);
    refreshStatus();
  }
}

$("run").addEventListener("click", () => runProcessing(() => {
  const form = new FormData();
  picked.forEach((file) => form.append("files", file, file.name));
  form.append("threshold", $("threshold").value);
  form.append("delete_mode", deleteMode());
  return api("/api/process", { method: "POST", body: form });
}));

$("run-folder").addEventListener("click", () => runProcessing(() => api("/api/process-folder", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ threshold: Number($("threshold").value), delete_mode: deleteMode() }),
})));

$("copy-message").addEventListener("click", async () => {
  const button = $("copy-message");
  try {
    await navigator.clipboard.writeText($("message").textContent);
    button.textContent = "Скопировано";
  } catch (e) {
    const range = document.createRange();
    range.selectNodeContents($("message"));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    button.textContent = "Текст выделен — Ctrl+C";
  }
  setTimeout(() => { button.textContent = "Скопировать"; }, 2500);
});

// ============================================================ разметка
const label = { name: null, tool: "cloud", drawing: false, last: null, dirty: false, labeled: new Set(), fillPoints: null, smartBusy: false };
const imageCanvas = $("canvas-image");
const labelCanvas = $("canvas-label");
const labelCtx = labelCanvas.getContext("2d");
const TOOL_COLORS = { cloud: "rgb(255,0,0)", clear: "rgb(0,0,255)" };

// История мазков. Одно нажатие кисти — один шаг: его можно отменить (Ctrl+Z)
// и вернуть обратно (Ctrl+X). Храним сами линии, а не копии холста, — так память не растёт.
const history = { base: null, steps: [], index: 0, savedIndex: 0, current: null };

function setLabelStatus(text, isError) {
  const box = $("label-status");
  box.textContent = text;
  box.classList.toggle("is-error", Boolean(isError));
}

function loadPicture(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Не удалось показать картинку"));
    img.src = src;
  });
}

async function loadImageList() {
  const list = $("image-list");
  try {
    const data = await api("/api/label/images");
    list.replaceChildren();
    label.labeled = new Set(data.images.filter((item) => item.labeled).map((item) => item.name));
    if (!data.images.length) {
      list.appendChild(el("li", "none", "Снимков пока нет. Добавьте их кнопкой выше."));
      return;
    }
    data.images.forEach((item) => {
      const button = el("button", item.name === label.name ? "is-active" : "");
      button.appendChild(el("span", "", item.name));
      if (item.labeled) button.appendChild(el("span", "tag", "размечен"));
      button.addEventListener("click", () => openImage(item.name));
      const li = el("li"); li.appendChild(button); list.appendChild(li);
    });
  } catch (e) {
    list.replaceChildren(el("li", "none", e.message));
  }
}

async function openImage(name) {
  if (label.dirty && !window.confirm("Мазки на текущем снимке не сохранены. Перейти к другому снимку?")) return;
  setLabelStatus("Открываю снимок…");
  try {
    const picture = await loadPicture("/api/label/image/" + encodeURIComponent(name) + "?t=" + Date.now());
    imageCanvas.width = labelCanvas.width = picture.naturalWidth;
    imageCanvas.height = labelCanvas.height = picture.naturalHeight;
    imageCanvas.getContext("2d").drawImage(picture, 0, 0);
    labelCtx.clearRect(0, 0, labelCanvas.width, labelCanvas.height);
    labelCtx.globalCompositeOperation = "source-over";
    history.base = null;
    if (label.labeled.has(name)) {
      const saved = await loadPicture("/api/label/mask/" + encodeURIComponent(name) + "?t=" + Date.now());
      labelCtx.drawImage(saved, 0, 0);
      history.base = saved;
    }
    history.steps = [];
    history.index = 0;
    history.savedIndex = 0;
    history.current = null;
    label.fillPoints = null;
    label.drawing = false;
    const fitWidth = 0.7 * window.innerHeight * picture.naturalWidth / picture.naturalHeight;
    $("stage").style.maxWidth = Math.round(Math.max(260, Math.min(900, fitWidth))) + "px";
    label.name = name;
    label.dirty = false;
    $("label-empty").hidden = true;
    $("label-work").hidden = false;
    $("show-prediction").checked = false;
    $("prediction").hidden = true;
    setLabelStatus("Проведите кистью по облакам и по местам без облаков, потом сохраните разметку.");
    loadImageList();
  } catch (e) {
    setLabelStatus(e.message, true);
  }
}

document.querySelectorAll(".tool").forEach((button) => {
  button.addEventListener("click", () => {
    label.tool = button.dataset.tool;
    document.querySelectorAll(".tool").forEach((b) => {
      b.classList.toggle("is-active", b === button);
      b.setAttribute("aria-checked", String(b === button));
    });
  });
});

function canvasPoint(event) {
  const rect = labelCanvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * labelCanvas.width / rect.width;
  const y = (event.clientY - rect.top) * labelCanvas.height / rect.height;
  return {
    // Pointer capture keeps receiving movement after the pointer leaves the canvas;
    // clamp it to the image edge so strokes and selection contours never escape it.
    x: Math.max(0, Math.min(labelCanvas.width, x)),
    y: Math.max(0, Math.min(labelCanvas.height, y)),
  };
}

function drawSegment(segment) {
  labelCtx.lineWidth = segment.width;
  labelCtx.lineCap = "round";
  labelCtx.lineJoin = "round";
  if (segment.tool === "erase") {
    labelCtx.globalCompositeOperation = "destination-out";
    labelCtx.strokeStyle = "rgb(0,0,0)";
  } else {
    labelCtx.globalCompositeOperation = "source-over";
    labelCtx.strokeStyle = TOOL_COLORS[segment.tool];
  }
  labelCtx.beginPath();
  labelCtx.moveTo(segment.from.x, segment.from.y);
  labelCtx.lineTo(segment.to.x + 0.01, segment.to.y + 0.01);
  labelCtx.stroke();
}

function stroke(from, to) {
  const scale = labelCanvas.width / labelCanvas.getBoundingClientRect().width;
  const segment = {
    from,
    to,
    width: Math.max(1.5, Number($("brush").value) * scale),
    tool: label.tool,
  };
  drawSegment(segment);
  if (history.current) history.current.push(segment);
  label.dirty = true;
}

// ---------- заливка области по свободному контуру ----------
function isFillMode() {
  return $("fill-mode").checked && label.tool !== "erase";
}

// ---------- умная заливка: клик по облаку, границы берём у сети ----------
function isSmartFill() {
  return $("smart-fill").checked && label.tool !== "erase";
}

function isCloudAreaFill() {
  return $("cloud-only-fill").checked;
}

function isOppositeColor(data, offset, tool) {
  if (data[offset + 3] === 0) return false;
  const red = data[offset] > data[offset + 2] + 24;
  if (tool === "clear") return red; // синяя заливка не перекрывает красные метки
  return red; // красная заливка может перекрасить синее, но сохраняет старые красные метки
}

// Красная заливка заменяет синие метки; красные сохраняются как границы.
// Синяя заливка не перекрашивает красные метки.
function paintMask(maskCanvas, tool) {
  const width = labelCanvas.width;
  const height = labelCanvas.height;
  const maskCtx = maskCanvas.getContext("2d");
  const source = maskCtx.getImageData(0, 0, width, height).data;
  const destination = labelCtx.getImageData(0, 0, width, height);
  const pixels = destination.data;
  for (let i = 0; i < source.length; i += 4) {
    const sourceAlpha = source[i + 3] / 255;
    if (!sourceAlpha || isOppositeColor(pixels, i, tool)) continue;
    const destinationAlpha = pixels[i + 3] / 255;
    const outputAlpha = sourceAlpha + destinationAlpha * (1 - sourceAlpha);
    for (let channel = 0; channel < 3; channel += 1) {
      pixels[i + channel] = Math.round((source[i + channel] * sourceAlpha
        + pixels[i + channel] * destinationAlpha * (1 - sourceAlpha)) / outputAlpha);
    }
    pixels[i + 3] = Math.round(outputAlpha * 255);
  }
  labelCtx.putImageData(destination, 0, 0);
}

function commitAction(action, message) {
  history.steps.length = history.index;
  history.steps.push([action]);
  history.index = history.steps.length;
  redrawHistory();
  updateDirty();
  setLabelStatus(message + " Ctrl+Z — отменить.");
}

async function smartFillAt(point) {
  if (!label.name || label.smartBusy) return;
  const name = label.name;
  const tool = label.tool;
  label.smartBusy = true;
  setLabelStatus("Спрашиваю сеть…");
  try {
    const data = await api("/api/label/region", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        x: Math.round(point.x),
        y: Math.round(point.y),
        tool,
        tolerance: Number($("smart-tolerance").value) / 100,
      }),
    });
    if (label.name !== name) return;              // снимок успели сменить, пока сеть думала
    if (!data.found) {
      setLabelStatus(data.message || "Здесь нечего закрашивать — попробуйте другую точку.", true);
      return;
    }
    const picture = await loadPicture(data.image);
    if (label.name !== name) return;
    history.steps.length = history.index;         // новая заливка стирает «отменённое» будущее
    history.steps.push([{ type: "region", image: picture, tool }]);
    history.index = history.steps.length;
    redrawHistory();
    updateDirty();
    const share = (100 * data.pixels / data.total).toFixed(1).replace(".", ",");
    setLabelStatus("Сеть с учётом границ цвета закрасила область: " + data.pixels + " пикселей (" + share + " % снимка). Ctrl+Z — отменить.");
  } catch (e) {
    if (label.name === name) setLabelStatus(e.message, true);
  } finally {
    label.smartBusy = false;
  }
}

async function cloudAreaFill(points) {
  if (!label.name || label.smartBusy) return;
  const name = label.name;
  const contour = points.map((point) => ({ x: Math.round(point.x), y: Math.round(point.y) }));
  label.smartBusy = true;
  setLabelStatus("ИИ ищет только облака внутри контура…");
  try {
    const data = await api("/api/label/cloud-area", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, points: contour }),
    });
    if (label.name !== name) return;
    if (!data.found) {
      setLabelStatus(data.message || "ИИ не нашёл облаков внутри контура.", true);
      return;
    }
    const picture = await loadPicture(data.image);
    if (label.name !== name) return;
    commitAction({ type: "region", image: picture, tool: "cloud" },
      "ИИ выделил только облака внутри замкнутого контура.");
    const share = (100 * data.pixels / data.total).toFixed(1).replace(".", ",");
    setLabelStatus("ИИ выделил облака: " + data.pixels + " пикселей (" + share + " % снимка). Ctrl+Z — отменить.");
  } catch (e) {
    if (label.name === name) setLabelStatus(e.message, true);
  } finally {
    label.smartBusy = false;
  }
}

function drawFill(points, tool) {
  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = labelCanvas.width;
  maskCanvas.height = labelCanvas.height;
  const maskCtx = maskCanvas.getContext("2d");
  maskCtx.strokeStyle = TOOL_COLORS[tool];
  maskCtx.fillStyle = TOOL_COLORS[tool];
  maskCtx.lineWidth = 2;
  maskCtx.lineJoin = "round";
  maskCtx.beginPath();
  maskCtx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i += 1) maskCtx.lineTo(points[i].x, points[i].y);
  // Замыкаем контур автоматически: открытая дуга заполняется до хорды.
  maskCtx.closePath();
  maskCtx.fill();
  maskCtx.stroke();
  paintMask(maskCanvas, tool);
}

function drawAction(action) {
  if (action.type === "fill") {
    drawFill(action.points, action.tool);
  } else if (action.type === "region") {
    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = labelCanvas.width;
    maskCanvas.height = labelCanvas.height;
    maskCanvas.getContext("2d").drawImage(action.image, 0, 0, labelCanvas.width, labelCanvas.height);
    paintMask(maskCanvas, action.tool);
  } else if (action.type === "paint-all-blue") {
    labelCtx.clearRect(0, 0, labelCanvas.width, labelCanvas.height);
    labelCtx.drawImage(action.image, 0, 0);
  } else if (action.type === "erase-all") {
    labelCtx.clearRect(0, 0, labelCanvas.width, labelCanvas.height);
  } else {
    drawSegment(action);
  }
}

// Два режима заливки не должны работать одновременно.
$("fill-mode").addEventListener("change", (e) => {
  if (e.target.checked) {
    $("smart-fill").checked = false;
    $("cloud-only-fill").checked = false;
  }
});
$("smart-fill").addEventListener("change", (e) => {
  if (e.target.checked) {
    $("fill-mode").checked = false;
    $("cloud-only-fill").checked = false;
  }
});
$("cloud-only-fill").addEventListener("change", (e) => {
  if (e.target.checked) {
    $("fill-mode").checked = false;
    $("smart-fill").checked = false;
  }
});
$("smart-tolerance").addEventListener("input", (e) => {
  $("smart-tolerance-value").value = e.target.value + "%";
  $("smart-tolerance-value").textContent = e.target.value + "%";
});

function setLabelOpacity(value, persist = false) {
  const opacity = Math.max(0, Math.min(100, Number(value) || 0));
  $("label-opacity").value = String(opacity);
  $("label-opacity-value").value = opacity + "%";
  $("label-opacity-value").textContent = opacity + "%";
  document.documentElement.style.setProperty("--label-alpha", String(opacity / 100));
  if (persist) {
    try { localStorage.setItem("cloud-label-opacity", String(opacity)); } catch (_) { /* storage may be disabled */ }
  }
}

let savedOpacity = 45;
try {
  const storedOpacity = localStorage.getItem("cloud-label-opacity");
  if (storedOpacity !== null && Number.isFinite(Number(storedOpacity))) savedOpacity = Number(storedOpacity);
} catch (_) { /* storage may be disabled */ }
setLabelOpacity(savedOpacity);
$("label-opacity").addEventListener("input", (e) => setLabelOpacity(e.target.value, true));

function drawFillPreview(from, to, tool = label.tool) {
  const scale = labelCanvas.width / labelCanvas.getBoundingClientRect().width;
  labelCtx.globalCompositeOperation = "source-over";
  labelCtx.strokeStyle = TOOL_COLORS[tool];
  labelCtx.lineWidth = Math.max(1.5, 2 * scale);
  labelCtx.lineCap = "round";
  labelCtx.lineJoin = "round";
  labelCtx.beginPath();
  labelCtx.moveTo(from.x, from.y);
  labelCtx.lineTo(to.x + 0.01, to.y + 0.01);
  labelCtx.stroke();
}

function completeFill() {
  const points = label.fillPoints || [];
  if (points.length < 3) {
    redrawHistory();
    setLabelStatus("Слишком короткий контур — заливка не сделана.", true);
    return;
  }
  if (isCloudAreaFill()) {
    const contour = points.map((point) => ({ ...point }));
    redrawHistory();
    cloudAreaFill(contour);
    return;
  }
  const action = { type: "fill", points, tool: label.tool };
  history.steps.length = history.index;    // новая заливка стирает «отменённое» будущее
  history.steps.push([action]);
  history.index = history.steps.length;
  redrawHistory();
  updateDirty();
  setLabelStatus("Область закрашена; концы контура автоматически соединены. Ctrl+Z — отменить.");
}

function endDrawing(cancel) {
  if (!label.drawing) return;
  label.drawing = false;
  if (label.fillPoints) {
    if (cancel) {
      label.fillPoints = null;
      redrawHistory();
      return;
    }
    completeFill();
    label.fillPoints = null;
    return;
  }
  if (history.current && history.current.length) {
    history.steps.length = history.index;   // новый мазок стирает «отменённое» будущее
    history.steps.push(history.current);
    history.index = history.steps.length;
    updateDirty();
  }
  history.current = null;
}

labelCanvas.addEventListener("pointerdown", (e) => {
  if (!label.name) return;
  const point = canvasPoint(e);
  if (isSmartFill()) {                            // один щелчок — сеть сама находит область
    smartFillAt(point);
    return;
  }
  labelCanvas.setPointerCapture(e.pointerId);
  label.drawing = true;
  label.last = point;
  if (isFillMode() || isCloudAreaFill()) {
    label.fillPoints = [point];
  } else {
    history.current = [];
    stroke(label.last, label.last);
  }
});
labelCanvas.addEventListener("pointermove", (e) => {
  if (!label.drawing) return;
  const point = canvasPoint(e);
  if (label.fillPoints) {
    const prev = label.fillPoints[label.fillPoints.length - 1];
    if (Math.hypot(point.x - prev.x, point.y - prev.y) >= 1.5) {
      label.fillPoints.push(point);
      drawFillPreview(prev, point, isCloudAreaFill() ? "cloud" : label.tool);
    }
  } else {
    stroke(label.last, point);
  }
  label.last = point;
});
labelCanvas.addEventListener("pointerup", () => endDrawing(false));
labelCanvas.addEventListener("pointercancel", () => endDrawing(true));

function redrawHistory() {
  labelCtx.globalCompositeOperation = "source-over";
  labelCtx.clearRect(0, 0, labelCanvas.width, labelCanvas.height);
  if (history.base) labelCtx.drawImage(history.base, 0, 0);
  for (let i = 0; i < history.index; i += 1) history.steps[i].forEach(drawAction);
  labelCtx.globalCompositeOperation = "source-over";
}

function updateDirty() {
  label.dirty = history.index !== history.savedIndex;
}

function undoStroke() {
  if (!label.name || label.drawing || !history.index) return false;
  history.index -= 1;
  redrawHistory();
  updateDirty();
  setLabelStatus("Мазок отменён. Ctrl+X вернёт его обратно.");
  return true;
}

function redoStroke() {
  if (!label.name || label.drawing || history.index >= history.steps.length) return false;
  history.index += 1;
  redrawHistory();
  updateDirty();
  setLabelStatus("Отменённый мазок возвращён.");
  return true;
}

$("fill-all-blue").addEventListener("click", () => {
  if (!label.name || label.drawing) return;
  const image = document.createElement("canvas");
  image.width = labelCanvas.width;
  image.height = labelCanvas.height;
  const imageCtx = image.getContext("2d");
  const pixels = labelCtx.getImageData(0, 0, labelCanvas.width, labelCanvas.height);
  for (let i = 0; i < pixels.data.length; i += 4) {
    const isRed = pixels.data[i + 3] > 0 && pixels.data[i] > pixels.data[i + 2] + 24;
    if (isRed) continue;
    pixels.data[i] = 0;
    pixels.data[i + 1] = 0;
    pixels.data[i + 2] = 255;
    pixels.data[i + 3] = 255;
  }
  imageCtx.putImageData(pixels, 0, 0);
  commitAction({ type: "paint-all-blue", image }, "Весь снимок закрашен синим, красные маркеры сохранены.");
});

$("erase-all").addEventListener("click", () => {
  if (!label.name || label.drawing) return;
  commitAction({ type: "erase-all" }, "Все маркеры стёрты с холста.");
});

document.addEventListener("keydown", (e) => {
  if (!label.name || panels.label.hidden || !(e.ctrlKey || e.metaKey)) return;
  const target = e.target;
  if (target && target.closest && (target.closest("input, textarea, select") || target.isContentEditable)) return;
  // e.code не зависит от раскладки: Ctrl+Z работает и на русской клавиатуре.
  const code = e.code || "";
  const key = (e.key || "").toLowerCase();
  const isZ = code === "KeyZ" || key === "z" || key === "я";
  const isX = code === "KeyX" || key === "x" || key === "ч";
  const isY = code === "KeyY" || key === "y" || key === "н";
  let handled = false;
  if (isZ && !e.shiftKey) handled = undoStroke();
  else if (isX || isY || (isZ && e.shiftKey)) handled = redoStroke();
  if (handled) e.preventDefault();
});

$("show-prediction").addEventListener("change", async (e) => {
  const picture = $("prediction");
  if (!e.target.checked) { picture.hidden = true; return; }
  setLabelStatus("Спрашиваю сеть…");
  try {
    const data = await api("/api/label/prediction/" + encodeURIComponent(label.name));
    picture.src = data.overlay;
    picture.hidden = false;
    setLabelStatus("Сеть считает, что облачность здесь " + percentText(data.cloud_percent) + ". Малиновым закрашено то, что она приняла за облака. Поправьте её мазками там, где она ошиблась.");
  } catch (err) {
    e.target.checked = false;
    setLabelStatus(err.message, true);
  }
});

$("transfer-clouds").addEventListener("click", async () => {
  if (!label.name || label.smartBusy) return;
  const name = label.name;
  label.smartBusy = true;
  setLabelStatus("Переношу на холст только облака, которые видит сеть…");
  try {
    const data = await api("/api/label/prediction-clouds/" + encodeURIComponent(name));
    if (label.name !== name) return;
    if (!data.cloud_pixels) {
      setLabelStatus("Сеть не нашла облаков на этом снимке.");
      return;
    }
    const picture = await loadPicture(data.image);
    if (label.name !== name) return;
    commitAction({ type: "region", image: picture, tool: "cloud" },
      "Облака, распознанные сетью, перенесены на холст.");
    setLabelStatus("Перенесено облаков: " + data.cloud_pixels + " пикселей. Синие области не добавлялись; Ctrl+Z — отменить.");
  } catch (err) {
    if (label.name === name) setLabelStatus(err.message, true);
  } finally {
    label.smartBusy = false;
  }
});

$("save-label").addEventListener("click", async () => {
  if (!label.name) return;
  try {
    const data = await api("/api/label/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: label.name, image: labelCanvas.toDataURL("image/png") }),
    });
    label.dirty = false;
    history.savedIndex = history.index;
    const total = data.cloud_pixels + data.clear_pixels;
    setLabelStatus(total
      ? "Разметка сохранена: облако — " + data.cloud_pixels + " пикселей, не облако — " + data.clear_pixels + ". Теперь можно дообучить сеть."
      : "На холсте нет мазков — разметка для этого снимка удалена.");
    loadImageList();
  } catch (e) {
    setLabelStatus(e.message, true);
  }
});

$("real-files").addEventListener("change", async (e) => {
  if (!e.target.files.length) return;
  const form = new FormData();
  Array.from(e.target.files).forEach((file) => form.append("files", file, file.name));
  try {
    await api("/api/label/upload", { method: "POST", body: form });
    e.target.value = "";
    loadImageList();
  } catch (err) {
    window.alert(err.message);
  }
});

async function trainRequest(path, busyText, button, idleText) {
  const log = $("train-log");
  log.hidden = false;
  log.textContent = busyText;
  $("train").disabled = true;
  $("reset-model").disabled = true;
  $("train-unet").disabled = true;
  button.textContent = busyText;
  try {
    const data = await api(path, { method: "POST" });
    const info = data.info;
    const held = info.holdout || {};
    const pct = (value) => (value * 100).toFixed(1).replace(".", ",") + " %";
    const summary = [];
    if (info.promoted === false) {
      summary.push("Новая модель НЕ принята: на снимках, которых она не видела, она не лучше базовой. Прежняя модель осталась.");
    } else {
      summary.push("Модель: " + (info.model || "обычная сеть") + ", с вашей разметкой (снимков: " + info.real_images.length + ", пикселей: " + info.real_pixels + ")");
    }
    if (held.candidate) {
      summary.push("Проверка на снимках, которых сеть не видела (" + info.holdout_images.length + " шт.): совпадение облаков "
        + pct(held.candidate.cloud_iou) + (held.reference ? ", " + held.reference_source + ": " + pct(held.reference.cloud_iou) : ""));
    }
    summary.push("Проверьте результат: включите «Показать, что видит сеть» или запустите python evaluate.py");
    const lines = (data.log || []).concat([""], summary);
    log.textContent = lines.join("\n");
    if ($("show-prediction").checked) $("show-prediction").dispatchEvent(new Event("change"));
  } catch (e) {
    log.textContent = e.message;
  } finally {
    $("train").disabled = false;
    $("reset-model").disabled = false;
    $("train-unet").disabled = false;
    button.textContent = idleText;
    refreshStatus();
  }
}

$("train").addEventListener("click", () => trainRequest("/api/train", "Обучаю сеть… одна-две минуты", $("train"), "Дообучить сеть"));
$("train-unet").addEventListener("click", () => trainRequest("/api/train-unet", "Обучаю U-Net… это займёт 10–15 минут", $("train-unet"), "Обучить U-Net (долго)"));
$("reset-model").addEventListener("click", () => {
  if (window.confirm("Вернуть базовую модель? Ваша разметка останется, но сеть забудет дообучение.")) {
    trainRequest("/api/model/reset", "Возвращаю базовую модель…", $("reset-model"), "Вернуть базовую модель");
  }
});

window.addEventListener("beforeunload", (e) => {
  if (label.dirty) { e.preventDefault(); e.returnValue = ""; }
});

refreshStatus();
