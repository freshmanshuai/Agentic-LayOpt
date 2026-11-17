import { createLayoutViewport } from "./three_viewport.js?v=20260909-session-provenance";

const DEFAULT_SOLVER = "genetic_algorithm";
const SOLVER_LABELS = {
  genetic_algorithm: "Genetic Algorithm",
};

const AGENTS = ["intent", "designer", "validator", "evaluator"];
const state = {
  backendUrl: "/api/rough-design/run",
  streamUrl: "/api/agentic-layopt/stream",
  viewportAckUrl: "/api/agentic-layopt/viewport-ack",
  sessionId: `web-${Date.now()}`,
  viewport: null,
  latestLayout: null,
  pendingLayout: null,
  latestResponse: null,
  latestSolverReport: null,
  latestValidation: null,
  latestEvaluation: null,
  latestSemanticIntent: null,
  clarificationOptions: [],
  clarificationQuestions: [],
  sessionAssetLabels: new Map(),
  latestViewportReceipt: null,
  capturedWorkspace: false,
  recordingOnly: false,
  changeComparisonBeforeLayout: null,
  changeComparisonAwaitingClarification: false,
  currentChangeComparison: null,
  streamEvents: [],
  activeRequestController: null,
  selectedInstanceId: null,
  solverSelectionTouched: false,
  runStartedAt: null,
  runElapsedS: 0,
  elapsedTimer: null,
  agentStartedAt: {},
  semanticOverrides: {},
  currentIntentType: "unknown_or_unsupported",
  flags: {
    showBoundary: true,
    showOverlap: true,
    showFlow: true,
    showGrid: true,
    showModels: true,
    showWalls: true,
    showAisles: true,
    showMaintenance: true,
  },
};

const el = {
  shell: document.querySelector("#app-shell"),
  backendState: document.querySelector("#backend-state"),
  viewportState: document.querySelector("#viewport-state"),
  viewportMode: document.querySelector("#viewport-mode"),
  requestState: document.querySelector("#request-state"),
  taskTypeState: document.querySelector("#task-type-state"),
  engineeringContext: document.querySelector("#engineering-context"),
  workflowStatus: document.querySelector("#workflow-status"),
  qualityStatus: document.querySelector("#quality-status"),
  elapsed: document.querySelector("#elapsed-time"),
  prompt: document.querySelector("#prompt-input"),
  run: document.querySelector("#run-button"),
  clear: document.querySelector("#clear-button"),
  save: document.querySelector("#save-button"),
  export: document.querySelector("#export-button"),
  restore: document.querySelector("#restore-button"),
  toggleLeft: document.querySelector("#toggle-left"),
  toggleRight: document.querySelector("#toggle-right"),
  canvas: document.querySelector("#three-viewport"),
  empty: document.querySelector("#viewport-empty"),
  layoutStatus: document.querySelector("#layout-status"),
  viewportInteractionHint: document.querySelector("#viewport-interaction-hint"),
  selection: document.querySelector("#selection-output"),
  propertyContext: document.querySelector("#property-context"),
  chatLog: document.querySelector("#chat-log"),
  semantic: document.querySelector("#semantic-output"),
  trace: document.querySelector("#trace-output"),
  recentTrace: document.querySelector("#trace-recent"),
  validation: document.querySelector("#validation-output"),
  validationDetail: document.querySelector("#validation-detail-output"),
  evaluation: document.querySelector("#evaluation-output"),
  evaluationDetail: document.querySelector("#evaluation-detail-output"),
  solver: document.querySelector("#solver-output"),
  flpIr: document.querySelector("#flp-ir-output"),
  rag: document.querySelector("#rag-output"),
  clarification: document.querySelector("#clarification-output"),
  clarificationBlock: document.querySelector("#clarification-block"),
  solverSelect: document.querySelector("#solver-select"),
  solverUsed: document.querySelector("#solver-used"),
  cameraMode: document.querySelector("#camera-mode"),
  resetCamera: document.querySelector("#reset-camera"),
  showBoundary: document.querySelector("#show-boundary"),
  showOverlap: document.querySelector("#show-overlap"),
  showFlow: document.querySelector("#show-flow"),
  showGrid: document.querySelector("#show-grid"),
  showModels: document.querySelector("#show-models"),
  showWalls: document.querySelector("#show-walls"),
  showAisles: document.querySelector("#show-aisles"),
  showMaintenance: document.querySelector("#show-maintenance"),
  showFlowLabel: document.querySelector("#show-flow-label"),
  hudSolver: document.querySelector("#hud-solver"),
  hudProgress: document.querySelector("#hud-progress"),
  hudCurrentLabel: document.querySelector("#hud-current-label"),
  hudBestLabel: document.querySelector("#hud-best-label"),
  hudCurrentScore: document.querySelector("#hud-current-score"),
  hudBestScore: document.querySelector("#hud-best-score"),
  hudViolations: document.querySelector("#hud-violations"),
  hudElapsed: document.querySelector("#hud-elapsed"),
  drawer: document.querySelector("#details-drawer"),
  drawerToggle: document.querySelector("#drawer-toggle"),
  drawerStatus: document.querySelector("#drawer-status"),
  centerStage: document.querySelector(".center-stage"),
  objectiveCount: document.querySelector("#objective-count"),
  toastRegion: document.querySelector("#toast-region"),
  workflowErrorActions: document.querySelector("#workflow-error-actions"),
  retry: document.querySelector("#retry-button"),
  viewErrorLog: document.querySelector("#view-error-log"),
  parameters: {
    population: document.querySelector("#parameter-population"),
    mutation: document.querySelector("#parameter-mutation"),
    iterations: document.querySelector("#parameter-iterations"),
    seed: document.querySelector("#parameter-seed"),
    timeout: document.querySelector("#parameter-timeout"),
  },
  requirements: {
    panel: document.querySelector("#requirement-review"),
    state: document.querySelector("#requirements-state"),
    intent: document.querySelector("#requirement-intent"),
    siteWidth: document.querySelector("#requirement-site-width"),
    siteDepth: document.querySelector("#requirement-site-depth"),
    assets: document.querySelector("#requirement-assets"),
    constraints: document.querySelector("#requirement-constraints"),
    objectives: document.querySelector("#requirement-objectives"),
    missing: document.querySelector("#requirement-missing"),
    apply: document.querySelector("#apply-requirements"),
    sources: {
      intent: document.querySelector("#source-intent"),
      site: document.querySelector("#source-site"),
      assets: document.querySelector("#source-assets"),
      constraints: document.querySelector("#source-constraints"),
      objectives: document.querySelector("#source-objectives"),
    },
  },
  clarificationReply: document.querySelector("#clarification-reply"),
  clarificationSend: document.querySelector("#clarification-send"),
  agents: Object.fromEntries(AGENTS.map((name) => [name, document.querySelector(`#agent-${name}`)])),
  agentSteps: Object.fromEntries(AGENTS.map((name) => [name, document.querySelector(`#agent-step-${name}`)])),
  topSteps: Object.fromEntries(AGENTS.map((name) => [name, document.querySelector(`#top-step-${name}`)])),
  agentSummaries: Object.fromEntries(AGENTS.map((name) => [name, document.querySelector(`#agent-${name}-summary`)])),
  agentTimes: Object.fromEntries(AGENTS.map((name) => [name, document.querySelector(`#agent-${name}-time`)])),
  objectives: {
    distance: document.querySelector("#objective-distance"),
    utilization: document.querySelector("#objective-utilization"),
    adjacency: document.querySelector("#objective-adjacency"),
    process: document.querySelector("#objective-process"),
    change: document.querySelector("#objective-change"),
  },
};

const examplePrompt =
  "Generate an initial layout for a 30 m by 20 m workshop with two engine lathes and one C-frame pressure tester. Keep equipment inside the boundary and avoid overlaps.";

init();

function init() {
  el.prompt.placeholder = examplePrompt;
  el.run.addEventListener("click", onPrimaryAction);
  el.clear.addEventListener("click", clearWorkspace);
  el.save.addEventListener("click", saveWorkspace);
  el.export.addEventListener("click", exportWorkspace);
  el.restore.addEventListener("click", restoreWorkspace);
  el.toggleLeft.addEventListener("click", () => el.shell.classList.toggle("left-collapsed"));
  el.toggleRight.addEventListener("click", () => el.shell.classList.toggle("right-collapsed"));
  el.drawerToggle.addEventListener("click", () => toggleDrawer());
  document.querySelector("#open-trace").addEventListener("click", () => openDrawer("trace"));
  el.retry.addEventListener("click", () => {
    if (!state.activeRequestController) runRoughDesign();
  });
  el.viewErrorLog.addEventListener("click", () => openDrawer("validation"));

  bindTabs(".context-tab", "contextTab", ".context-pane", "contextPane");
  bindTabs(".subtab", "qualityTab", ".quality-pane", "qualityPane");
  bindTabs(".drawer-tab", "drawerTab", ".drawer-pane", "drawerPane");

  for (const eventName of ["keydown", "keyup", "keypress"]) {
    el.prompt.addEventListener(eventName, (event) => event.stopPropagation());
  }
  el.prompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      onPrimaryAction();
    }
  });
  el.requirements.apply.addEventListener("click", applyRequirementEdits);
  el.clarificationSend.addEventListener("click", submitClarificationReply);
  el.solverSelect.value = DEFAULT_SOLVER;
  el.solverSelect.addEventListener("change", () => {
    state.solverSelectionTouched = true;
    renderSelectedSolver();
    updateHud();
  });
  for (const option of document.querySelectorAll(".prompt-option")) {
    option.addEventListener("toggle", () => {
      if (!option.open) return;
      for (const peer of document.querySelectorAll(".prompt-option")) {
        if (peer !== option) peer.open = false;
      }
    });
  }
  for (const objective of Object.values(el.objectives)) {
    objective.addEventListener("change", updateObjectiveCount);
  }
  for (const toggle of [
    el.showBoundary,
    el.showOverlap,
    el.showFlow,
    el.showGrid,
    el.showModels,
    el.showWalls,
    el.showAisles,
    el.showMaintenance,
  ]) {
    toggle.addEventListener("change", updateVisualizationFlags);
  }
  el.cameraMode.addEventListener("change", () => state.viewport?.setCameraMode?.(el.cameraMode.value));
  el.resetCamera.addEventListener("click", () => state.viewport?.resetCamera?.());

  updateObjectiveCount();
  setRequestStatus("IDLE");
  updateDrawerStatus();
  checkBackend();
  configureSolvers();
  configureViewport();
  resetConversation();
  el.prompt.disabled = false;
  el.run.disabled = false;
  void restoreCapturedWorkspaceFromUrl();
}

function onPrimaryAction() {
  if (!state.activeRequestController) {
    runRoughDesign();
    return;
  }
  state.activeRequestController.abort();
  state.activeRequestController = null;
  setBusy(false);
  stopElapsedTimer();
  setRequestStatus("STOPPED");
  el.workflowStatus.textContent = "Stopped by user";
  showToast("Solver stopped.");
}

async function checkBackend() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const health = await response.json();
    state.recordingOnly = health.recording_only === true;
    if (state.recordingOnly) {
      el.prompt.disabled = true;
      el.run.disabled = true;
    }
    setChip(el.backendState, state.recordingOnly ? "Recorded viewer" : "Backend connected", "ok");
  } catch (_error) {
    setChip(el.backendState, "Backend offline", "error");
  }
}

async function configureSolvers() {
  try {
    const response = await fetch("/api/solver/config");
    if (!response.ok) return;
    const config = await response.json();
    const availability = new Map((config.solvers || []).map((item) => [item.name, item]));
    for (const option of el.solverSelect.options) {
      const item = availability.get(option.value);
      option.disabled = Boolean(item && !item.available);
      const label = solverLabel(option.value);
      option.textContent = item && !item.available ? `${label} (unavailable)` : label;
    }
    if (!state.solverSelectionTouched) {
      const configuredDefault = config.default_solver || DEFAULT_SOLVER;
      const option = [...el.solverSelect.options]
        .find((item) => item.value === configuredDefault && !item.disabled);
      el.solverSelect.value = option ? configuredDefault : DEFAULT_SOLVER;
    }
  } catch (_error) {
    // The selected local default remains usable when discovery is unavailable.
  }
  renderSelectedSolver();
}

async function configureViewport() {
  try {
    const response = await fetch("/api/viewport/config");
    const config = await response.json();
    if (!response.ok || !config.available || !config.manifest_url) {
      throw new Error(config.error || `Viewport config HTTP ${response.status}`);
    }
    setChip(el.viewportState, "Viewport loading", "muted");
    state.viewport = await createLayoutViewport({
      canvas: el.canvas,
      emptyState: el.empty,
      manifestUrl: config.manifest_url,
      onSelect: onViewportSelected,
    });
    window.__agenticLayOptDebug = () => ({
      ...(state.viewport?.debugState() || {}),
      sessionId: state.sessionId,
      latestLayout: state.latestLayout,
    });
    setChip(el.viewportState, "Viewport connected", "ok");
    el.viewportMode.textContent = "";
    updateVisualizationFlags();
    if (state.latestLayout) sendLayoutToViewport(state.latestLayout);
  } catch (error) {
    console.error("Three.js viewport initialization failed", error);
    setChip(el.viewportState, "Viewport unavailable", "error");
    el.viewportMode.textContent = "";
    el.empty.hidden = false;
    el.empty.textContent = `Viewport initialization failed: ${error.message}`;
  }
}

async function runRoughDesign() {
  if (state.recordingOnly || state.capturedWorkspace) {
    showToast("This is a recorded view. Clear it and use the normal server for a new run.");
    return;
  }
  if (!state.changeComparisonAwaitingClarification) {
    state.changeComparisonBeforeLayout = state.latestLayout
      ? structuredClone(state.latestLayout)
      : null;
  }
  state.changeComparisonAwaitingClarification = false;
  state.capturedWorkspace = false;
  const prompt = el.prompt.value.trim();
  if (!prompt) {
    el.prompt.focus();
    return;
  }

  appendMessage("user", prompt);
  el.prompt.value = "";
  state.streamEvents = [];
  state.latestResponse = null;
  state.latestSolverReport = null;
  state.latestValidation = null;
  state.latestEvaluation = null;
  state.latestSemanticIntent = null;
  state.clarificationOptions = [];
  state.clarificationQuestions = [];
  state.agentStartedAt = {};
  AGENTS.forEach((name) => { el.agentTimes[name].textContent = ""; });
  state.runElapsedS = 0;
  const requestController = new AbortController();
  state.activeRequestController = requestController;
  startElapsedTimer();
  setBusy(true);
  setAgentStates("running", "pending", "pending", "pending");
  setRequestStatus("RUNNING");
  el.workflowStatus.textContent = "Understanding requirement";
  el.layoutStatus.textContent = "Waiting for a feasible layout";
  setText(el.trace, "Waiting for agent events.");
  setText(el.solver, "Waiting for solver metrics.");
  renderValidation(null);
  renderEvaluation(null);
  renderClarification([]);
  updateHud();

  try {
    const request = buildRequest(prompt);
    state.semanticOverrides = {};
    const response = await fetch(state.streamUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(request),
      signal: requestController.signal,
    });
    const contentType = response.headers.get("Content-Type") || "";
    if (response.ok && contentType.includes("text/event-stream")) {
      await readEventStream(response);
      setChip(el.backendState, "Backend connected", "ok");
      return;
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const fallback = await fetch(state.backendUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(request),
      signal: requestController.signal,
    });
    const payload = await fallback.json();
    if (!fallback.ok) throw new Error(payload.error || `HTTP ${fallback.status}`);
    applyBackendResponse(payload.layout_response || payload);
    setChip(el.backendState, "Backend connected", "ok");
  } catch (error) {
    if (error.name !== "AbortError") {
      setAgentStates("failed", "skipped", "skipped", "skipped");
      el.workflowStatus.textContent = "Request failed";
      setRequestStatus("BACKEND_ERROR");
      renderValidation({ status: "REQUEST_FAILED", errors: [{ message: error.message }], warnings: [] });
      setChip(el.backendState, "Backend error", "error");
      openDrawer("validation");
    }
  } finally {
    if (state.activeRequestController === requestController) {
      state.activeRequestController = null;
      setBusy(false);
      stopElapsedTimer();
    }
  }
}

async function readEventStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() || "";
    for (const chunk of chunks) {
      const event = parseSseChunk(chunk);
      if (event) await handleStreamEvent(event);
    }
  }
  if (buffer.trim()) {
    const event = parseSseChunk(buffer);
    if (event) await handleStreamEvent(event);
  }
}

function parseSseChunk(chunk) {
  const dataLine = chunk.split("\n").find((line) => line.startsWith("data: "));
  if (!dataLine) return null;
  try {
    return JSON.parse(dataLine.slice(6));
  } catch (error) {
    console.warn("Invalid SSE event", error);
    return null;
  }
}

async function handleStreamEvent(event) {
  recordStreamEvent(event);
  updateLiveProgress(event);
  const summary = event.summary || event.status || "running";

  if (event.event === "intent_started") {
    markAgentStarted("intent", summary);
    el.workflowStatus.textContent = "Understanding requirement";
  }
  if (event.event === "intent_done") {
    markAgentCompleted("intent", summary);
    renderSemanticIntent(event.payload?.semantic_intent, event.payload?.intent);
  }
  if (event.event === "rag_done") {
    state.clarificationOptions = publicClarificationOptions(
      event.payload?.clarification_options,
    );
  }
  if (event.event === "design_started") {
    markAgentStarted("designer", summary);
    el.workflowStatus.textContent = "Generating layout";
  }
  if (event.event === "design_done") {
    const design = event.payload?.design || {};
    const skipped = event.status === "skipped" || String(design.design_status || "").includes("skipped");
    if (skipped) {
      setAgentStatus("designer", "skipped", "Not required");
      el.agentSummaries.designer.textContent = skippedAgentMessage("designer", state.currentIntentType);
    } else {
      markAgentCompleted("designer", friendlyStatus(design.design_status || "completed"));
    }
    state.latestSolverReport = design.solver_report || state.latestSolverReport;
    state.pendingLayout = design.rough_layout || null;
    if (!skipped) renderSolverReport(state.latestSolverReport);
    renderRawArtifacts(design);
  }
  if (event.event === "validation_started") {
    markAgentStarted("validator", summary);
    el.workflowStatus.textContent = "Checking constraints";
  }
  if (event.event === "validation_done") {
    const validationFrame = event.payload?.validation || {};
    const validation = withSolverReady(validationFrame.validation_report);
    const tone = event.status === "skipped"
      ? "skipped"
      : validation?.errors?.length
      ? "failed"
      : validation?.warnings?.length ? "warning" : "completed";
    if (tone === "skipped") {
      setAgentStatus("validator", "skipped", "Not required");
      el.agentSummaries.validator.textContent = skippedAgentMessage("validator", state.currentIntentType);
    } else {
      markAgentCompleted("validator", friendlyStatus(validation?.status || "completed"), tone);
    }
    renderValidation(validation);
    const clarificationOptions = publicClarificationOptions(
      event.payload?.clarification_options,
    );
    const clarificationQuestions = validationFrame.repair?.clarification_questions || [];
    const needsClarification = validation?.status === "NEED_CLARIFICATION"
      || clarificationQuestions.length > 0;
    if (clarificationOptions.length) {
      state.clarificationOptions = clarificationOptions;
    }
    state.clarificationQuestions = clarificationQuestions;
    if (needsClarification) {
      renderClarification(
        clarificationQuestions,
        clarificationOptions.length
          ? clarificationOptions
          : state.clarificationOptions,
      );
    } else {
      state.clarificationOptions = [];
      state.clarificationQuestions = [];
      renderClarification([]);
    }
    if (["VALID_SOLVER_READY", "VALID_WITH_WARNINGS"].includes(validation?.status) && state.pendingLayout) {
      state.latestLayout = state.pendingLayout;
    }
    state.pendingLayout = null;
  }
  if (event.event === "evaluation_started") {
    markAgentStarted("evaluator", summary);
    el.workflowStatus.textContent = "Evaluating layout";
  }
  if (event.event === "evaluation_done") {
    const evaluation = event.payload?.evaluation;
    const skipped = event.status === "skipped" || evaluation?.evaluation_status === "not_evaluated";
    if (skipped) {
      setAgentStatus("evaluator", "skipped", "Not evaluated");
      el.agentSummaries.evaluator.textContent = skippedAgentMessage("evaluator", state.currentIntentType);
    } else {
      markAgentCompleted("evaluator", friendlyStatus(evaluation?.evaluation_status || "completed"));
    }
    renderEvaluation(evaluation);
  }
  if (event.event === "feedback_generated") {
    setAgentStatus("designer", "warning", `Revision ${(event.iteration || 0) + 1} queued`);
  }
  if (event.event === "loop_done") {
    const report = event.payload?.loop_report || {};
    setRequestStatus(event.payload?.status || event.status, report.termination_reason);
  }
  if (event.event === "viewport_command") {
    const command = event.viewport_command || {};
    if (command.method === "RenderLayoutJson") {
      state.latestLayout = command.payload;
      const comparisonStage = changeComparisonStage(state.currentIntentType);
      const changeComparison = comparisonStage && state.changeComparisonBeforeLayout
        ? {
          stage: comparisonStage,
          beforeLayout: state.changeComparisonBeforeLayout,
        }
        : null;
      state.currentChangeComparison = changeComparison;
      await sendLayoutToViewport(
        command.payload,
        command.render_token || null,
        event.session_id || state.sessionId,
        {
          turnId: command.turn_id ?? event.turn_id ?? null,
          layoutSha256: command.layout_sha256 || null,
          changeComparison,
        },
      );
      state.changeComparisonBeforeLayout = null;
      state.changeComparisonAwaitingClarification = false;
    }
  }
  if (event.event === "turn_done" || event.event === "turn_blocked") {
    const status = event.payload?.status || event.summary || event.status;
    const reason = event.payload?.loop_report?.termination_reason;
    setRequestStatus(status, reason);
    el.workflowStatus.textContent = event.event === "turn_blocked"
      ? (status === "NEED_CLARIFICATION" ? "Waiting for information" : "Task blocked")
      : "Workflow complete";
    if (event.event === "turn_blocked" && status === "NEED_CLARIFICATION") {
      state.changeComparisonAwaitingClarification = true;
    } else if (event.event === "turn_blocked" || event.event === "turn_done") {
      state.changeComparisonBeforeLayout = null;
      state.changeComparisonAwaitingClarification = false;
    }
    state.latestResponse = event.payload || null;
    renderRawArtifacts(event.payload || {});
    if (status === "NEED_CLARIFICATION") {
      const terminalOptions = publicClarificationOptions(
        event.payload?.clarification_options,
      );
      const terminalQuestions = event.payload?.clarification_questions || state.clarificationQuestions;
      if (terminalOptions.length) state.clarificationOptions = terminalOptions;
      state.clarificationQuestions = terminalQuestions;
      renderClarification(
        terminalQuestions,
        terminalOptions.length ? terminalOptions : state.clarificationOptions,
      );
    }
    appendAssistantMessage(event.payload?.assistant_message, event.payload);
    updateDrawerStatus();
    finishRequest();
  }
}

function withSolverReady(report) {
  if (!report) return report;
  return {
    ...report,
    solver_ready: ["VALID_SOLVER_READY", "VALID_WITH_WARNINGS"].includes(report.status),
  };
}

function changeComparisonStage(intentType) {
  const value = String(intentType || "").toLowerCase();
  if (value === "edit_layout" || value === "edit") return "edit";
  if (value === "constraint_repair" || value === "repair") return "repair";
  return null;
}

function buildRequest(prompt) {
  const overrides = state.semanticOverrides || {};
  const hardConstraints = overrides.constraints?.length ? overrides.constraints : ["no_overlap"];
  const solverOptions = selectedSolverOptions();
  return {
    session_id: state.sessionId,
    solver: el.solverSelect.value || DEFAULT_SOLVER,
    raw_user_message: prompt,
    engineering_context: el.engineeringContext.value,
    structured_constraints: {
      site_width_m: overrides.site?.width_m || 0,
      site_depth_m: overrides.site?.depth_m || 0,
      hard_constraints: hardConstraints,
      objectives: selectedObjectives(),
    },
    selected_assets: (overrides.asset_mentions || []).map((item) => ({
      model_asset_id: item.model_asset_id || item.surface_text,
    })),
    scene_context: {
      active_layout_id: state.latestLayout?.layout_id || "",
      selected_instance_ids: state.selectedInstanceId ? [state.selectedInstanceId] : [],
      semantic_overrides: overrides,
      solver_options: solverOptions,
      time_limit_s: optionalNumber(el.parameters.timeout.value),
    },
  };
}

function selectedSolverOptions() {
  const solver = el.solverSelect.value;
  const seed = optionalNumber(el.parameters.seed.value);
  const iterations = optionalNumber(el.parameters.iterations.value);
  const options = {};
  if (seed != null) options.seed = seed;
  if (solver === "genetic_algorithm") {
    const population = optionalNumber(el.parameters.population.value);
    const mutation = optionalNumber(el.parameters.mutation.value);
    if (population != null) options.population_size = population;
    if (mutation != null) options.mutation_rate = mutation;
    if (iterations != null) options.generations = iterations;
  }
  return options;
}

function selectedObjectives() {
  const objectives = [];
  if (el.objectives.distance.checked) objectives.push("minimize_material_handling_distance");
  if (el.objectives.utilization.checked) objectives.push("maximize_space_utilization");
  if (el.objectives.adjacency.checked) objectives.push("maximize_adjacency_satisfaction");
  if (el.objectives.process.checked) objectives.push("maximize_process_order_consistency");
  if (el.objectives.change.checked) objectives.push("minimize_layout_change");
  return objectives;
}

function applyBackendResponse(response) {
  state.latestResponse = response;
  state.latestLayout = response.rough_layout || state.latestLayout;
  state.latestSolverReport = response.solver_report || null;
  setRequestStatus(response.status || "UNKNOWN", response.loop_report?.termination_reason);
  el.workflowStatus.textContent = response.status === "NEED_CLARIFICATION" ? "Waiting for information" : "Workflow complete";
  applyAgentTrace(response.agent_trace || []);
  renderSemanticIntent(response.semantic_intent, response.intent);
  renderValidation(response.validation_report);
  renderEvaluation(response.evaluation);
  renderSolverReport(response.solver_report);
  const clarificationOptions = publicClarificationOptions(
    response.clarification_options,
  );
  const clarificationQuestions = response.clarification_questions || [];
  if (clarificationOptions.length) state.clarificationOptions = clarificationOptions;
  state.clarificationQuestions = clarificationQuestions;
  if (response.status === "NEED_CLARIFICATION" || clarificationQuestions.length) {
    renderClarification(
      clarificationQuestions,
      clarificationOptions.length ? clarificationOptions : state.clarificationOptions,
    );
  } else {
    state.clarificationOptions = [];
    state.clarificationQuestions = [];
    renderClarification([]);
  }
  renderRawArtifacts(response);
  state.selectedInstanceId = null;
  updateSelection(null);
  if (state.latestLayout) sendLayoutToViewport(state.latestLayout);
  appendAssistantMessage(response.assistant_message, response);
  updateDrawerStatus();
}

function recordStreamEvent(event) {
  state.streamEvents.push({
    event: event.event || "message",
    agent: event.agent_name || "unknown",
    status: event.status || "",
    summary: event.summary || event.payload?.status || "",
    displaySummary: friendlyEventSummary(event),
    iteration: event.iteration ?? null,
    timestamp: new Date().toISOString(),
    raw: event,
  });
  renderEventLog();
}

function renderEventLog() {
  if (!state.streamEvents.length) {
    setText(el.trace, "No agent events yet.");
    el.recentTrace.innerHTML = "<li>Submit a task to start the workflow.</li>";
    return;
  }
  const lines = state.streamEvents.map((item, index) => {
    const status = item.status ? ` [${item.status}]` : "";
    const summary = item.summary ? ` - ${item.summary}` : "";
    const iteration = item.iteration != null ? ` (iteration ${item.iteration})` : "";
    return `${index + 1}. ${item.agent}: ${item.event}${iteration}${status}${summary}`;
  });
  setText(el.trace, state.streamEvents.map((item) => JSON.stringify(item.raw, null, 2)).join("\n\n"));
  el.recentTrace.replaceChildren();
  for (const item of state.streamEvents.slice(-3).reverse()) {
    const li = document.createElement("li");
    li.textContent = item.displaySummary;
    el.recentTrace.append(li);
  }
}

function appendAssistantMessage(text, response = {}) {
  if (typeof text === "string" && text.trim()) {
    appendMessage("assistant", text.trim(), { rich: true });
  }
  const evidence = response.assistant_message_evidence;
  if (!evidence) return;
  const source = response.assistant_message_source === "llm"
    ? `Live LLM response · ${evidence.provider || "configured provider"} · ${evidence.model || "configured model"} · ${formatNumber(evidence.latency_ms)} ms`
    : `Disclosed fallback response · ${response.assistant_message_error || evidence.status || "fallback"}`;
  const adapter = response.adapter_id ? ` · Case Contract Adapter: ${response.adapter_id}` : "";
  appendMessage("assistant", `${source}${adapter}`, { tone: "evidence" });
}

function appendMessage(role, text, options = {}) {
  const article = document.createElement("article");
  article.className = `message ${role === "user" ? "user-message" : "assistant-message"}`;
  if (options.tone) article.classList.add(`${options.tone}-message`);
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "U" : "A";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (options.rich) {
    bubble.classList.add("rich-message");
    renderRichMessage(bubble, text);
  } else {
    bubble.textContent = text;
  }
  article.append(avatar, bubble);
  el.chatLog.append(article);
  el.chatLog.scrollTop = el.chatLog.scrollHeight;
}

function resetConversation() {
  el.chatLog.replaceChildren();
}

function renderRichMessage(container, value) {
  let list = null;
  for (const sourceLine of String(value || "").replaceAll("\r", "").split("\n")) {
    const line = sourceLine.trim();
    if (!line) {
      list = null;
      continue;
    }
    const heading = line.match(/^(?:#{1,3}\s+(.+?)|\*\*(.+?)\*\*)\s*$/);
    if (heading) {
      list = null;
      const title = document.createElement("h3");
      title.textContent = (heading[1] || heading[2]).replace(/:$/, "");
      container.append(title);
      continue;
    }
    const bullet = line.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      if (!list) {
        list = document.createElement("ul");
        container.append(list);
      }
      const item = document.createElement("li");
      appendInlineEmphasis(item, bullet[1]);
      list.append(item);
      continue;
    }
    list = null;
    const paragraph = document.createElement("p");
    appendInlineEmphasis(paragraph, line);
    container.append(paragraph);
  }
}

function appendInlineEmphasis(container, value) {
  for (const part of String(value).split(/(\*\*[^*]+\*\*)/g)) {
    if (!part) continue;
    if (part.startsWith("**") && part.endsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = part.slice(2, -2);
      container.append(strong);
    } else {
      container.append(document.createTextNode(part));
    }
  }
}

function applyAgentTrace(trace) {
  if (!trace.length) {
    return;
  }
  for (const item of trace) {
    const agentText = (item.agent || "").toLowerCase();
    const name = AGENTS.find((candidate) => agentText.includes(candidate === "intent" ? "intent" : candidate));
    if (!name) continue;
    const rawStatus = item.decision || item.status || "completed";
    const status = normalizeAgentStatus(rawStatus);
    setAgentStatus(name, status, friendlyAgentStatus(status, rawStatus));
    if (item.runtime_s != null) el.agentTimes[name].textContent = `${formatNumber(item.runtime_s)}s`;
  }
  setText(el.trace, trace.map((item) => JSON.stringify(item)).join("\n"));
}

function renderSemanticIntent(frame, intent) {
  const safe = frame || {};
  state.latestSemanticIntent = safe;
  const intentType = safe.intent_type || intent?.intent || intent?.task_type || "unknown_or_unsupported";
  state.currentIntentType = intentType;
  const site = safe.site?.width_m && safe.site?.depth_m
    ? `${formatNumber(safe.site.width_m)}m × ${formatNumber(safe.site.depth_m)}m`
    : "Unresolved";
  const assetMentions = safe.asset_mentions || [];
  rememberSessionAssetLabels(assetMentions);
  const assets = state.capturedWorkspace && state.capturedEquipmentNames?.length
    ? state.capturedEquipmentNames
    : assetMentions.map(displayAssetMention).filter(Boolean);
  const assetUnitCount = equipmentUnitCount(assetMentions);
  const constraints = (safe.constraints || []).map((item) => item.constraint_type).filter(Boolean);
  const objectives = (safe.objectives || []).map((item) => item.objective_type).filter(Boolean);
  const missing = safe.missing_information || [];

  el.semantic.replaceChildren();
  addKv("Source", safe.source || "unknown");
  addKv("Intent", intentType);
  addKv("Confidence", safe.confidence != null ? formatNumber(safe.confidence) : "unknown");
  addKv("Site", site);
  addKv("Assets", assets.join(", ") || "unresolved");
  addKv("Constraints", constraints.join(", ") || "no_overlap");
  addKv("Objectives", objectives.join(", ") || "none");
  addKv("Missing", missing.join(", ") || "none");

  applyIntentRoute(intentType);
  el.taskTypeState.textContent = shortTaskType(intentType);
  el.taskTypeState.title = `Task: ${friendlyTaskType(intentType)}`;
  el.requirements.panel.hidden = false;
  el.requirements.intent.value = [...el.requirements.intent.options].some((option) => option.value === intentType)
    ? intentType
    : "unknown_or_unsupported";
  el.requirements.siteWidth.value = safe.site?.width_m ?? "";
  el.requirements.siteDepth.value = safe.site?.depth_m ?? "";
  el.requirements.assets.value = assets.join(", ");
  el.requirements.constraints.value = constraints.join(", ") || "no_overlap";
  el.requirements.objectives.value = objectives.join(", ");
  const unresolved = safe.unresolved_requirements || [];
  const missingLabels = unresolved.length
    ? unresolved.filter((item) => item.severity === "blocking").map((item) => friendlyRequirement(item.requirement_id, item.description))
    : missing;
  el.requirements.missing.textContent = missingLabels.length
    ? `Missing: ${missingLabels.join(" · ")}`
    : "";
  el.requirements.state.textContent = missingLabels.length ? "Needs review" : "Ready";
  setSourceBadge(el.requirements.sources.intent, "Detected", "detected");
  setSourceBadge(el.requirements.sources.site, sourceLabelFor(safe, "site"), sourceToneFor(safe, "site"));
  const assetSource = assetSourcePresentation(safe, assetMentions);
  setSourceBadge(
    el.requirements.sources.assets,
    assetSource.label,
    assetSource.tone,
  );
  setSourceBadge(el.requirements.sources.constraints, constraints.length ? sourceLabelFor(safe, "constraints") : "Default", constraints.length ? sourceToneFor(safe, "constraints") : "default");
  setSourceBadge(el.requirements.sources.objectives, objectives.length ? sourceLabelFor(safe, "objectives") : "Auto", objectives.length ? sourceToneFor(safe, "objectives") : "default");
  el.agentSummaries.intent.textContent = [
    friendlyTaskType(intentType),
    `${site}; ${assetUnitCount} equipment unit${assetUnitCount === 1 ? "" : "s"}`,
    constraints.length ? `${constraints.length} constraints` : "default no overlap",
    missingLabels.length ? `${missingLabels.length} item(s) need input` : "requirements ready",
  ].join(" · ");
  updateSiteProperties();
}

function equipmentUnitCount(assetMentions) {
  return (assetMentions || []).reduce((total, item) => {
    const quantity = Number(item?.quantity);
    return total + (Number.isFinite(quantity) && quantity > 0 ? Math.round(quantity) : 1);
  }, 0);
}

function addKv(key, value) {
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = key;
  dd.textContent = value;
  el.semantic.append(dt, dd);
}

function applyRequirementEdits() {
  if (state.activeRequestController) return;
  const width = optionalNumber(el.requirements.siteWidth.value);
  const depth = optionalNumber(el.requirements.siteDepth.value);
  const assets = splitList(el.requirements.assets.value);
  const constraints = splitList(el.requirements.constraints.value);
  const objectives = splitList(el.requirements.objectives.value);
  state.semanticOverrides = {
    intent_type: el.requirements.intent.value,
    site: width && depth ? { width_m: width, depth_m: depth, source_text: "confirmed in requirement review" } : null,
    asset_mentions: assets.map((value) => ({ surface_text: value })),
    constraints,
    objectives,
  };
  el.requirements.state.textContent = "Confirmed";
  setSourceBadge(el.requirements.sources.intent, "Confirmed", "user");
  if (width && depth) setSourceBadge(el.requirements.sources.site, "Confirmed", "user");
  if (assets.length) setSourceBadge(el.requirements.sources.assets, "Confirmed", "user");
  if (constraints.length) setSourceBadge(el.requirements.sources.constraints, "Confirmed", "user");
  if (objectives.length) setSourceBadge(el.requirements.sources.objectives, "Confirmed", "user");
  el.prompt.value = "Continue with the confirmed requirement changes.";
  runRoughDesign();
}

function submitClarificationReply() {
  const reply = el.clarificationReply.value.trim();
  if (!reply || state.activeRequestController) return;
  el.clarificationReply.value = "";
  el.prompt.value = reply;
  runRoughDesign();
}

function splitList(value) {
  return String(value || "").split(/[,;\n]/).map((item) => item.trim()).filter(Boolean);
}

function sourceEntryFor(frame, fieldPrefix) {
  return (frame.semantic_evidence || []).find((item) => String(item.field_path || "").startsWith(fieldPrefix));
}

function readableAssetMention(item) {
  const catalogName = [
    item?.catalog_name_en,
    item?.english_name,
    item?.display_name_en,
    item?.canonical_name_en,
  ].find(isReadableEnglishName);
  if (catalogName) {
    const quantity = Number(item?.quantity || 0);
    return quantity > 0 ? `${formatNumber(quantity)} × ${catalogName}` : catalogName;
  }
  return String(item?.surface_text || item?.equipment_type || "").trim();
}

function displayAssetMention(item) {
  const modelAssetId = String(item?.model_asset_id || "").trim();
  const sourceText = String(item?.surface_text || "").trim();
  if (modelAssetId && looksLikeAssetIdentifier(sourceText)) {
    const inheritedLabel = state.sessionAssetLabels.get(modelAssetId);
    if (inheritedLabel) return inheritedLabel;
  }
  return readableAssetMention(item);
}

function rememberSessionAssetLabels(assetMentions) {
  for (const item of assetMentions || []) {
    const modelAssetId = String(item?.model_asset_id || "").trim();
    const label = readableAssetMention(item);
    if (modelAssetId && label && !looksLikeAssetIdentifier(label)) {
      state.sessionAssetLabels.set(modelAssetId, label);
    }
  }
}

function isReadableEnglishName(value) {
  const label = String(value || "").trim();
  return Boolean(label && /[a-z]/i.test(label) && !looksLikeAssetIdentifier(label));
}

function assetSourcePresentation(frame, assetMentions) {
  const inherited = (assetMentions || []).some((item, index) => {
    const evidence = sourceEntryFor(frame, `asset_mentions[${index}]`);
    return item?.value_source === "authoritative_context"
      || evidence?.value_source === "authoritative_context";
  });
  if (inherited) return { label: "Inherited", tone: "inherited" };
  const resolvedFromCatalog = (assetMentions || []).some((item) => {
    const modelAssetId = String(item?.model_asset_id || "").trim();
    const surfaceText = String(item?.surface_text || "").trim();
    return modelAssetId && !surfaceText.includes(modelAssetId);
  });
  if (resolvedFromCatalog) return { label: "Catalog", tone: "catalog" };
  return {
    label: sourceLabelFor(frame, "asset_mentions"),
    tone: sourceToneFor(frame, "asset_mentions"),
  };
}

function looksLikeAssetIdentifier(value) {
  return /^(?:asset\/|model_asset_id\b)/i.test(String(value || "").trim());
}

function sourceLabelFor(frame, fieldPrefix) {
  const source = sourceEntryFor(frame, fieldPrefix)?.value_source;
  return {
    user_input: "User",
    existing_layout: "Existing layout",
    authoritative_context: "Inherited",
    model_inference: "Inferred",
    system_default: "Default",
  }[source] || "Missing";
}

function sourceToneFor(frame, fieldPrefix) {
  const source = sourceEntryFor(frame, fieldPrefix)?.value_source;
  if (source === "authoritative_context") return "inherited";
  if (source === "model_inference") return "inferred";
  if (source === "system_default") return "default";
  if (!source) return "missing";
  return "user";
}

function setSourceBadge(target, label, tone) {
  target.textContent = label;
  target.className = `source-badge ${tone || ""}`.trim();
}

function friendlyRequirement(requirementId, fallback) {
  return {
    missing_site_dimensions: "site width and depth",
    missing_equipment_requirements: "equipment or engineering domain",
    missing_active_layout: "an active layout",
    unsupported_design_intent: "a supported layout task",
  }[requirementId] || fallback || "additional information";
}

function applyIntentRoute(intentType) {
  const routes = {
    generate_feasible_layout: ["intent", "designer", "validator"],
    optimize_layout_from_scratch: ["intent", "designer", "validator", "evaluator"],
    optimize_existing_layout: ["intent", "designer", "validator", "evaluator"],
    edit_layout: ["intent", "designer", "validator"],
    constraint_repair: ["intent", "designer", "validator"],
    validate_layout: ["intent", "validator"],
    evaluate_layout: ["intent", "validator", "evaluator"],
  };
  const active = new Set(routes[intentType] || ["intent"]);
  for (const name of AGENTS) {
    if (name === "intent") continue;
    if (active.has(name)) {
      setAgentStatus(name, "pending", "Pending");
      el.agentSummaries[name].textContent = waitingMessage(name, intentType);
    } else {
      setAgentStatus(name, "skipped", "Not required");
      el.agentSummaries[name].textContent = skippedAgentMessage(name, intentType);
    }
  }
}

function waitingMessage(agent, intentType) {
  if (agent === "designer") return intentType === "edit_layout" ? "Waiting for an editable layout." : "Waiting for confirmed requirements.";
  if (agent === "validator") return "Waiting for the relevant layout artifact.";
  return "Waiting for a validated layout.";
}

function skippedAgentMessage(agent, intentType) {
  if (agent === "designer" && ["validate_layout", "evaluate_layout"].includes(intentType)) return "No design or solver run is needed for this task.";
  if (agent === "evaluator" && intentType === "validate_layout") return "Evaluation was not requested.";
  if (agent === "evaluator") return "No validated layout is available for evaluation.";
  if (agent === "validator" && intentType === "unknown_or_unsupported") return "Waiting for a clear task type.";
  return "This step is not required for the current task.";
}

function friendlyTaskType(intentType) {
  return {
    generate_feasible_layout: "Generate layout",
    optimize_layout_from_scratch: "Optimize new layout",
    optimize_existing_layout: "Optimize current layout",
    edit_layout: "Edit current layout",
    constraint_repair: "Repair layout",
    validate_layout: "Validate current layout",
    evaluate_layout: "Evaluate current layout",
    unknown_or_unsupported: "Clarify task",
  }[intentType] || "Clarify task";
}

function shortTaskType(intentType) {
  return {
    generate_feasible_layout: "Generate",
    optimize_layout_from_scratch: "Optimize new",
    optimize_existing_layout: "Optimize current",
    edit_layout: "Edit",
    constraint_repair: "Repair",
    validate_layout: "Validate",
    evaluate_layout: "Evaluate",
    unknown_or_unsupported: "Clarify",
  }[intentType] || "Clarify";
}

function renderValidation(report) {
  if (!report) {
    state.latestValidation = null;
    el.validation.textContent = "Validation will appear when a candidate layout is ready.";
    setText(el.validationDetail, "No validation details.");
    el.qualityStatus.textContent = "Not validated";
    updateDrawerStatus();
    return;
  }

  report = withSolverReady(report);
  state.latestValidation = report;
  if (["NEED_CLARIFICATION", "UNSUPPORTED_TASK"].includes(report.status)) {
    el.validation.textContent = report.status === "NEED_CLARIFICATION"
      ? "Validation is waiting for the missing task information."
      : "Validation is not applicable to this task.";
    el.qualityStatus.textContent = friendlyStatus(report.status);
    setText(el.validationDetail, JSON.stringify(report, null, 2));
    updateDrawerStatus();
    return;
  }

  const errors = report.errors || [];
  const warnings = report.warnings || [];
  const checks = collectConstraintChecks(report);
  el.validation.replaceChildren();
  const list = document.createElement("div");
  list.className = "constraint-list";
  for (const check of checks) {
    const row = document.createElement("div");
    row.className = `constraint-row ${check.tone}`;
    row.innerHTML = `<span class="constraint-icon">${check.tone === "pass" ? "✓" : check.tone === "fail" ? "!" : "·"}</span><span>${escapeHtml(check.label)}</span><strong>${escapeHtml(check.result)}</strong>`;
    if (check.tone === "fail") {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "Locate";
      button.addEventListener("click", () => state.viewport?.highlightViolations?.(report));
      row.lastElementChild.replaceWith(button);
    }
    list.append(row);
  }
  el.validation.append(list);
  el.qualityStatus.textContent = friendlyStatus(report.status || "unknown");
  setText(el.validationDetail, JSON.stringify(report, null, 2));
  el.agentSummaries.validator.textContent = `${friendlyStatus(report.status || "unknown")} · ${errors.length} errors · ${warnings.length} warnings`;
  updateHud();
  updateSiteProperties();
  updateDrawerStatus();
  if (errors.length) showToast(`${errors.length} validation error(s). Open details to inspect.`);
}

function collectConstraintChecks(report) {
  const errors = report.errors || [];
  const warnings = report.warnings || [];
  const haystack = [...errors, ...warnings]
    .map((item) => JSON.stringify(item).toLowerCase())
    .join(" ");
  return [
    ["Boundary", ["boundary", "site"]],
    ["No overlap", ["overlap"]],
    ["Avoid aisle", ["aisle", "non_placeable"]],
    ["Wall clearance", ["wall_clearance", "wall clearance"]],
    ["Maintenance clearance", ["maintenance", "clearance"]],
  ].map(([label, tokens]) => {
    const failed = errors.some((item) => tokens.some((token) => JSON.stringify(item).toLowerCase().includes(token)));
    const warned = warnings.some((item) => tokens.some((token) => JSON.stringify(item).toLowerCase().includes(token)));
    if (failed) return { label, tone: "fail", result: "Failed" };
    if (warned) return { label, tone: "warn", result: "Warning" };
    if (!haystack && !report.solver_ready) return { label, tone: "neutral", result: "Not reported" };
    return { label, tone: "pass", result: "Passed" };
  });
}

function renderEvaluation(evaluation) {
  state.latestEvaluation = evaluation || null;
  if (!evaluation) {
    el.evaluation.textContent = "Evaluation will appear after validation succeeds.";
    setText(el.evaluationDetail, "No evaluation details.");
    updateDrawerStatus();
    return;
  }

  const metrics = evaluation.metrics || {};
  const status = evaluation.evaluation_status || evaluation.status || "unknown";
  if (status === "not_evaluated") {
    el.evaluation.textContent = "Evaluation was not required or could not run because no validated layout was available.";
    setText(el.evaluationDetail, JSON.stringify(evaluation, null, 2));
    el.agentSummaries.evaluator.textContent = skippedAgentMessage("evaluator", state.currentIntentType);
    updateDrawerStatus();
    return;
  }
  const utilization = metrics.site_utilization ?? metrics.estimated_area_utilization;
  const overlapCount = metrics.overlap_violation_count ?? metrics.overlap_count ?? 0;
  const boundaryCount = metrics.boundary_violation_count ?? 0;
  const nonPlaceableCount = metrics.non_placeable_violation_count ?? 0;
  const clearanceCount = metrics.clearance_violation_count ?? 0;
  const rows = [
    ["Overall score", metrics.overall_score ?? metrics.objective_value],
    ["Space utilization", utilization],
    ["Flow distance", metrics.flow_distance],
    ["Process consistency", metrics.process_consistency ?? metrics.process_order_score],
    ["Maintenance access", metrics.maintenance_accessibility ?? metrics.maintenance_score],
    ["Accessibility", metrics.accessibility ?? metrics.accessibility_score],
    ["Placed equipment", metrics.placed_instances],
  ];
  el.evaluation.replaceChildren();
  if (evaluation.summary) {
    const summary = document.createElement("p");
    summary.className = "evaluation-summary";
    summary.textContent = evaluation.summary;
    el.evaluation.append(summary);
  }
  const list = document.createElement("div");
  list.className = "metric-list";
  for (const [label, value] of rows) {
    const row = document.createElement("div");
    row.className = "metric-row";
    row.innerHTML = `<span>${escapeHtml(label)}</span><strong>${escapeHtml(formatMetric(value))}</strong>`;
    list.append(row);
  }
  el.evaluation.append(list);
  setText(el.evaluationDetail, JSON.stringify(evaluation, null, 2));
  el.agentSummaries.evaluator.textContent = `${status} · utilization ${formatMetric(utilization)} · ${overlapCount + boundaryCount + nonPlaceableCount + clearanceCount} violations`;
  updateHud();
  updateSiteProperties();
  updateDrawerStatus();
}

function renderSolverReport(report) {
  state.latestSolverReport = report || null;
  if (!report) {
    setText(el.solver, "No solver report.");
    el.solverUsed.textContent = "Not run";
    updateHud();
    return;
  }

  const violations = report.violations || [];
  const used = report.method === "user_constraint_repair" ? "constraint_repair" : report.solver_used || report.solver || el.solverSelect.value;
  el.solverUsed.textContent = solverShortLabel(used);
  el.solverUsed.title = solverLabel(used);
  const metrics = solverMetricRows(report, used);
  setText(el.solver, metrics.map(([label, value]) => `${label}: ${value}`).join("\n"));
  el.agentSummaries.designer.textContent = `Designer called ${solverLabel(used)} · ${friendlyStatus(report.status || "unknown")} · ${solverHeadline(report, used)}`;
  updateHud();
  updateDrawerStatus();
}

function solverMetricRows(report, solver) {
  const base = [
    ["Requested", report.solver_requested ? solverLabel(report.solver_requested) : "Automatic"],
    ["Method used", solverLabel(solver)],
    ["Status", report.status || "unknown"],
    [report.method === "user_constraint_repair" ? "Repair translation (m)" : "Objective value", formatMetric(report.objective_value)],
    ["Runtime", report.runtime_s == null ? "n/a" : `${formatMetric(report.runtime_s)}s`],
    ["Seed", report.seed ?? "n/a"],
    ["Violations", report.violation_count ?? report.violations?.length ?? "n/a"],
  ];
  const objectives = report.objective_values || {};
  if (objectives.minimize_agv_transport_cost != null) {
    base.push(["J1 AGV transport cost index/day", formatMetric(objectives.minimize_agv_transport_cost)]);
  }
  if (objectives.minimize_weighted_flow_distance != null) {
    base.push(["J1 batch-weighted handling distance", formatMetric(objectives.minimize_weighted_flow_distance)]);
  }
  if (objectives.minimize_relocation_effort != null) {
    base.push(["J2 relocation effort index", formatMetric(objectives.minimize_relocation_effort)]);
  } else if (objectives.minimize_high_flow_distance_excess != null) {
    base.push(["J2 high-flow distance excess", formatMetric(objectives.minimize_high_flow_distance_excess)]);
  }
  if (report.pareto_front) {
    base.push(["Nondominated candidates", report.pareto_front.length], ["Selected candidate", report.selected_candidate_id || "None"]);
  }
  if (solver === "genetic_algorithm") {
    base.push(
      ["Generation", report.generations_completed ?? "n/a"],
      ["Population", report.population_size ?? "n/a"],
      ["Best fitness", formatMetric(report.best_fitness ?? report.objective_value)],
      ["Feasible layouts", report.feasible_layouts ?? report.feasible_count ?? "n/a"],
    );
  } else {
    base.push(["Method", report.method || "n/a"], ["Iterations", report.iterations_completed ?? "n/a"]);
  }
  return base;
}

function solverHeadline(report, solver) {
  if (solver === "genetic_algorithm") return `generation ${report.generations_completed ?? "n/a"}`;
  return report.method || "layout generated";
}

function publicClarificationOptions(value) {
  return (Array.isArray(value) ? value : [])
    .map((option) => ({
      surfaceText: String(option?.surface_text || option?.surfaceText || "").trim(),
      choices: (Array.isArray(option?.choices) ? option.choices : [])
        .map((choice) => String(
          typeof choice === "string" ? choice : choice?.display_label || "",
        ).trim())
        .filter(Boolean),
    }))
    .filter((option) => option.surfaceText && option.choices.length);
}

function renderClarification(questions, options = []) {
  const publicOptions = publicClarificationOptions(options);
  const visible = (Array.isArray(questions) && questions.length > 0)
    || publicOptions.length > 0;
  el.clarificationBlock.hidden = !visible;
  el.clarification.replaceChildren();
  if (!visible) return;
  if (Array.isArray(questions) && questions.length) {
    const list = document.createElement("ul");
    for (const question of questions) {
      const item = document.createElement("li");
      item.textContent = question;
      list.append(item);
    }
    el.clarification.append(list);
  }
  for (const option of publicOptions) {
    const group = document.createElement("div");
    group.className = "clarification-choice-group";
    const title = document.createElement("p");
    title.className = "clarification-choice-title";
    title.textContent = `Catalog choices for "${option.surfaceText}"`;
    group.append(title);
    const choices = document.createElement("div");
    choices.className = "clarification-choice-list";
    for (const label of option.choices) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "clarification-choice";
      button.dataset.clarificationChoice = label;
      button.textContent = label;
      button.addEventListener("click", () => {
        el.clarificationReply.value = `Use ${label}.`;
        el.clarificationReply.focus();
      });
      choices.append(button);
    }
    group.append(choices);
    el.clarification.append(group);
  }
  activateContextTab("workflow");
}

function renderRawArtifacts(payload) {
  const source = payload || {};
  const flpIr = source.flp_ir || source.design?.flp_ir || source.layout_response?.flp_ir;
  const rag = source.rag_evidence || source.evidence_package || source.evidence
    || source.design?.evidence_package;
  if (flpIr) setText(el.flpIr, JSON.stringify(flpIr, null, 2));
  if (rag) setText(el.rag, JSON.stringify(rag, null, 2));
  if (source.validation_report) setText(el.validationDetail, JSON.stringify(source.validation_report, null, 2));
  if (source.evaluation) setText(el.evaluationDetail, JSON.stringify(source.evaluation, null, 2));
}

async function sendLayoutToViewport(
  layout,
  renderToken = null,
  receiptSessionId = null,
  receiptContext = {},
) {
  if (!state.viewport) return;
  if (!layout) {
    state.viewport.clear();
    el.layoutStatus.textContent = "No feasible layout generated";
    updateSelection(null);
    return;
  }
  updateVisualizationFlags();
  try {
    const receipt = await state.viewport.render(layout, {
      turnId: receiptContext.turnId ?? null,
      layoutSha256: receiptContext.layoutSha256 || null,
      layoutSnapshot: layout,
      renderToken,
      changeComparison: receiptContext.changeComparison || null,
    });
    if (receipt?.render_status === "SUPERSEDED") return;
    if (!renderToken) {
      state.latestViewportReceipt = null;
      const rendered = receipt?.rendered_instance_ids?.length || 0;
      el.layoutStatus.textContent = state.capturedWorkspace
        ? `${rendered} equipment loaded from a captured accepted run · local view`
        : `${rendered} equipment rendered · local preview`;
      updateSiteProperties();
      return;
    }
    const verified = await acknowledgeViewportReceipt(
      receipt,
      receiptSessionId || state.sessionId,
    );
    state.latestViewportReceipt = verified;
    const rendered = verified.rendered_instance_count ?? receipt?.rendered_instance_ids?.length ?? 0;
    if (verified.status === "VERIFIED") {
      el.layoutStatus.textContent = `${rendered} equipment rendered · backend verified`;
    } else {
      el.layoutStatus.textContent = `${rendered} equipment rendered · ${verified.status.toLowerCase()}`;
    }
    updateSiteProperties();
  } catch (error) {
    console.error("Layout rendering failed", error);
    el.layoutStatus.textContent = "Layout visible but render receipt was not verified";
  }
}

async function acknowledgeViewportReceipt(receipt, sessionId) {
  const response = await fetch(state.viewportAckUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, receipt }),
  });
  const body = await response.json();
  if (!response.ok || !body.viewport_receipt) {
    throw new Error(body.error || `Viewport receipt HTTP ${response.status}`);
  }
  return body.viewport_receipt;
}

function updateVisualizationFlags() {
  state.flags = {
    showBoundary: el.showBoundary.checked,
    showOverlap: el.showOverlap.checked,
    showFlow: el.showFlow.checked,
    showGrid: el.showGrid.checked,
    showModels: el.showModels.checked,
    showWalls: el.showWalls.checked,
    showAisles: el.showAisles.checked,
    showMaintenance: el.showMaintenance.checked,
  };
  state.viewport?.setFlags(state.flags);
}

function clearWorkspace() {
  state.activeRequestController?.abort();
  state.activeRequestController = null;
  stopElapsedTimer();
  state.sessionId = `web-${Date.now()}`;
  state.latestLayout = null;
  state.pendingLayout = null;
  state.latestResponse = null;
  state.latestSolverReport = null;
  state.latestValidation = null;
  state.latestEvaluation = null;
  state.latestSemanticIntent = null;
  state.clarificationOptions = [];
  state.clarificationQuestions = [];
  state.sessionAssetLabels.clear();
  state.latestViewportReceipt = null;
  state.capturedWorkspace = false;
  state.capturedElapsedText = null;
  state.capturedEquipmentNames = [];
  el.prompt.disabled = state.recordingOnly;
  el.run.disabled = state.recordingOnly;
  el.prompt.placeholder = state.recordingOnly ? "Recorded session (read-only)." : examplePrompt;
  el.engineeringContext.disabled = state.recordingOnly;
  for (const control of el.requirements.panel.querySelectorAll("input, select, textarea, button")) {
    control.disabled = state.recordingOnly;
  }
  state.changeComparisonBeforeLayout = null;
  state.changeComparisonAwaitingClarification = false;
  state.currentChangeComparison = null;
  state.streamEvents = [];
  state.selectedInstanceId = null;
  state.runElapsedS = 0;
  el.prompt.value = "";
  el.engineeringContext.value = "";
  resetConversation();
  el.semantic.replaceChildren();
  for (const objective of Object.values(el.objectives)) objective.checked = false;
  for (const parameter of Object.values(el.parameters)) parameter.value = "";
  el.solverSelect.value = DEFAULT_SOLVER;
  state.solverSelectionTouched = false;
  setAgentStates("pending", "pending", "pending", "pending");
  AGENTS.forEach((name) => { el.agentTimes[name].textContent = ""; });
  setRequestStatus("IDLE");
  el.taskTypeState.textContent = "Task not identified";
  el.requirements.panel.hidden = true;
  el.workflowStatus.textContent = "Waiting";
  el.layoutStatus.textContent = "No layout generated";
  setText(el.trace, "No agent events yet.");
  setText(el.solver, "No solver report.");
  setText(el.flpIr, "No FLP-IR available.");
  setText(el.rag, "No RAG evidence available.");
  renderValidation(null);
  renderEvaluation(null);
  renderClarification([]);
  renderSelectedSolver();
  updateObjectiveCount();
  updateSelection(null);
  updateElapsedDisplay();
  updateHud();
  updateDrawerStatus();
  setBusy(false);
  state.viewport?.clear();
  el.prompt.focus();
}

function onViewportSelected(placement) {
  state.selectedInstanceId = placement?.instance_id || null;
  updateSelection(placement);
  activateContextTab("properties");
}

function updateSelection(selection) {
  if (!selection) {
    updateSiteProperties();
    return;
  }
  el.propertyContext.textContent = "Equipment";
  const dimensions = selection.dimensions_m || {};
  const footprint = selection.footprint || {};
  const rows = [
    ["Equipment", selection.name || selection.instance_id || "unknown"],
    ["Model asset", selection.source_model_asset_id || selection.model_asset_id || "unknown"],

    ["Model dimensions", formatDimensions(dimensions)],
    ["Layout envelope", formatFootprint(footprint)],
    ["Position", formatPosition(selection.position)],
    ["Rotation", `${formatMetric(selection.rotation?.yaw_deg)}°`],
    ["Locked", selection.locked || selection.fixed ? "Yes" : "No"],
    ["Maintenance space", formatObjectSummary(selection.maintenance_clearance_m || selection.maintenance_space)],
    ["Input ports", formatObjectSummary(selection.input_ports || selection.ports?.input)],
    ["Output ports", formatObjectSummary(selection.output_ports || selection.ports?.output)],
    ["Relations", formatObjectSummary(selection.relations || selection.relationship_constraints)],
  ];
  renderPropertyRows(rows);
}

function updateSiteProperties() {
  el.propertyContext.textContent = "Site";
  const layout = state.latestLayout || {};
  const site = layout.site || state.latestSemanticIntent?.site || {};
  const dimensions = site.dimensions_m || site;
  const metrics = state.latestEvaluation?.metrics || {};
  const violations = validationIssueCount();
  const rows = [
      ["Site size", dimensions.width_m && dimensions.depth_m
        ? `${formatNumber(dimensions.width_m)} m × ${formatNumber(dimensions.depth_m)} m`
        : "Unresolved"],
      ["Equipment", layout.placements?.length ?? state.latestSemanticIntent?.asset_mentions?.length ?? 0],
      ["Area utilization", formatMetric(metrics.site_utilization ?? metrics.estimated_area_utilization)],
      ["Layout score", formatMetric(metrics.overall_score ?? metrics.objective_value)],
      ["Constraint violations", violations],
    ];
  renderPropertyRows(rows);
}

function renderPropertyRows(rows) {
  el.selection.replaceChildren();
  const sheet = document.createElement("div");
  sheet.className = "property-grid";
  for (const [label, value] of rows) {
    const row = document.createElement("div");
    row.className = "property-row";
    const key = document.createElement("span");
    key.textContent = label;
    const data = document.createElement("strong");
    data.textContent = value == null ? "n/a" : String(value);
    row.append(key, data);
    sheet.append(row);
  }
  el.selection.append(sheet);
}

function setAgentStates(intent, designer, validator, evaluator) {
  const values = { intent, designer, validator, evaluator };
  for (const name of AGENTS) {
    const normalized = normalizeAgentStatus(values[name]);
    setAgentStatus(name, normalized, friendlyAgentStatus(normalized, values[name]));
  }
}

function setAgentStatus(name, status, label) {
  const normalized = normalizeAgentStatus(status);
  el.agents[name].textContent = label || normalized;
  el.agentSteps[name].dataset.status = normalized;
  el.topSteps[name].dataset.status = normalized;
  el.workflowErrorActions.hidden = !AGENTS.some((agent) => el.agentSteps[agent].dataset.status === "failed");
}

function normalizeAgentStatus(status) {
  const value = String(status || "pending").toLowerCase();
  if (["running", "started", "processing", "in_progress"].some((item) => value.includes(item))) return "running";
  if (["failed", "error", "invalid", "blocked"].some((item) => value.includes(item))) return "failed";
  if (["warning", "warn", "with_warnings"].some((item) => value.includes(item))) return "warning";
  if (["skipped", "not run", "not_run", "not_evaluated", "not_invoked", "not_required"].some((item) => value.includes(item))) return "skipped";
  if (["done", "completed", "valid", "evaluated", "feasible", "optimal", "ready"].some((item) => value.includes(item))) return "completed";
  return "pending";
}

function markAgentStarted(name, summary) {
  state.agentStartedAt[name] = performance.now();
  setAgentStatus(name, "running", "Running");
  el.agentSummaries[name].textContent = summary || "Processing backend data.";
}

function markAgentCompleted(name, summary, tone = "completed") {
  setAgentStatus(name, tone, friendlyAgentStatus(tone, summary));
  const started = state.agentStartedAt[name];
  if (started != null) el.agentTimes[name].textContent = `${((performance.now() - started) / 1000).toFixed(2)}s actual elapsed`;
}

function updateLiveProgress(event) {
  const payload = event.payload || {};
  const metrics = payload.metrics || payload.solver_metrics || payload.design?.solver_report || {};
  const hasSolverProgress = [
    "iteration",
    "iterations_completed",
    "generation",
    "generations_completed",
    "current_score",
    "best_score",
    "energy_value",
    "objective_value",
  ].some((field) => metrics[field] != null);
  if (!hasSolverProgress) return;
  const iteration = metrics.iteration ?? metrics.iterations_completed ?? metrics.generation ?? metrics.generations_completed;
  const total = metrics.total_iterations ?? metrics.max_iterations ?? metrics.maximum_iterations ?? metrics.total_generations;
  if (iteration != null) el.hudProgress.textContent = total ? `${iteration} / ${total}` : String(iteration);
  const current = metrics.current_score ?? metrics.energy_value ?? metrics.objective_value;
  const best = metrics.best_score ?? metrics.best_energy ?? metrics.best_fitness;
  if (current != null) el.hudCurrentScore.textContent = formatMetric(current);
  if (best != null) el.hudBestScore.textContent = formatMetric(best);
}

function updateHud() {
  const report = state.latestSolverReport;
  el.hudCurrentLabel.textContent = "Current";
  el.hudBestLabel.textContent = "Best";
  if (!report || !Object.keys(report).length) {
    const exactEdit = state.currentIntentType === "edit_layout" && state.latestEvaluation?.metrics?.objective_value != null;
    const agvObjective = (state.latestSemanticIntent?.objectives || []).some((o) => o.objective_type === "minimize_agv_transport_cost");
    el.hudSolver.textContent = exactEdit ? "Exact edit" : state.activeRequestController ? "Waiting" : "Not run";
    el.hudProgress.textContent = "-";
    el.hudCurrentScore.textContent = exactEdit ? formatMetric(state.latestEvaluation.metrics.objective_value) : "-";
    if (exactEdit) el.hudCurrentLabel.textContent = agvObjective ? "J1 AGV cost" : "Objective";
    el.hudBestScore.textContent = "-";
    const validationKnown = state.latestValidation
      && !["NOT_RUN", ""].includes(String(state.latestValidation.status || ""));
    el.hudViolations.textContent = validationKnown ? String(validationIssueCount()) : "-";
    updateElapsedDisplay();
    return;
  }
  const solver = report.method === "user_constraint_repair" ? "constraint_repair" : report.solver_used || report.solver || "unknown";
  el.hudSolver.textContent = solverLabel(solver);
  const progress = report.generations_completed ?? report.iterations_completed;
  el.hudProgress.textContent = progress != null ? String(progress) : "-";
  const current = report.energy_value ?? report.objective_value;
  const best = report.best_energy ?? report.best_fitness ?? report.best_score;
  el.hudCurrentScore.textContent = current != null ? formatMetric(current) : "-";
  el.hudBestScore.textContent = best != null ? formatMetric(best) : "-";
  const values = report.objective_values || {};
  const second = values.minimize_relocation_effort ?? values.minimize_high_flow_distance_excess;
  const first = values.minimize_agv_transport_cost ?? values.minimize_weighted_flow_distance;
  if (first != null && second != null) {
    el.hudCurrentLabel.textContent = values.minimize_agv_transport_cost != null ? "J1 AGV cost" : "J1";
    el.hudBestLabel.textContent = values.minimize_relocation_effort != null ? "J2 effort" : "J2 excess";
    el.hudCurrentScore.textContent = formatMetric(first);
    el.hudBestScore.textContent = formatMetric(second);
    el.hudProgress.textContent = `${report.pareto_front?.length ?? 0} Pareto`;
  }
  const violations = report.violation_count ?? report.violations?.length;
  const validationKnown = state.latestValidation
    && !["NOT_RUN", ""].includes(String(state.latestValidation.status || ""));
  el.hudViolations.textContent = violations != null
    ? String(violations)
    : validationKnown ? String(validationIssueCount()) : "-";
  updateElapsedDisplay();
}

function startElapsedTimer() {
  stopElapsedTimer();
  state.runStartedAt = performance.now();
  state.elapsedTimer = window.setInterval(() => {
    state.runElapsedS = (performance.now() - state.runStartedAt) / 1000;
    updateElapsedDisplay();
  }, 100);
}

function stopElapsedTimer() {
  if (state.runStartedAt != null) state.runElapsedS = (performance.now() - state.runStartedAt) / 1000;
  if (state.elapsedTimer != null) window.clearInterval(state.elapsedTimer);
  state.elapsedTimer = null;
  state.runStartedAt = null;
  updateElapsedDisplay();
}

function updateElapsedDisplay() {
  const text = state.capturedWorkspace && state.capturedElapsedText
    ? state.capturedElapsedText
    : `${state.runElapsedS.toFixed(1)}s`;
  el.elapsed.textContent = text;
  el.hudElapsed.textContent = text;
}

function setBusy(isBusy) {
  el.run.textContent = isBusy ? "Stop" : "Send";
  el.run.classList.toggle("stop", isBusy);
  el.solverSelect.disabled = isBusy;
}

function finishRequest() {
  const controller = state.activeRequestController;
  state.activeRequestController = null;
  setBusy(false);
  stopElapsedTimer();
  controller?.abort();
}

function renderSelectedSolver() {
  el.solverUsed.textContent = solverShortLabel(el.solverSelect.value);
  el.solverUsed.title = solverLabel(el.solverSelect.value);
}

function solverShortLabel(value) {
  return {
    genetic_algorithm: "GA",
    constraint_repair: "Repair",
  }[value] || "Auto";
}

function updateObjectiveCount() {
  const count = selectedObjectives().length;
  el.objectiveCount.textContent = count ? String(count) : "Auto";
}

function toggleDrawer(forceOpen) {
  const shouldOpen = forceOpen ?? el.drawer.classList.contains("collapsed");
  el.drawer.classList.toggle("collapsed", !shouldOpen);
  el.centerStage.classList.toggle("drawer-open", shouldOpen);
  el.drawerToggle.setAttribute("aria-expanded", String(shouldOpen));
}

function openDrawer(tab) {
  toggleDrawer(true);
  activateDrawerTab(tab);
}

function activateDrawerTab(tab) {
  const button = document.querySelector(`.drawer-tab[data-drawer-tab="${tab}"]`);
  button?.click();
}

function updateDrawerStatus() {
  const errors = state.latestValidation?.errors?.length || 0;
  const status = el.requestState.textContent || "Idle";
  el.drawerStatus.textContent = `${status} · ${errors} error${errors === 1 ? "" : "s"}`;
}

function setRequestStatus(status, reason) {
  let label = friendlyStatus(status);
  if (["VALID_SOLVER_READY", "VALID_WITH_WARNINGS"].includes(String(status))) {
    if (state.currentIntentType === "validate_layout") label = "Validation passed";
    else if (state.currentIntentType === "evaluate_layout") label = "Evaluation complete";
    else if (state.currentIntentType === "generate_feasible_layout") label = "Layout ready";
    else if (["optimize_layout_from_scratch", "optimize_existing_layout"].includes(state.currentIntentType)) label = "Optimization complete";
  }
  el.requestState.textContent = label;
  el.requestState.title = reason ? `${label}: ${friendlyReason(reason)}` : label;
  updateDrawerStatus();
}

function friendlyStatus(status) {
  const value = String(status || "unknown").toLowerCase();
  const exact = {
    idle: "Idle",
    running: "Running",
    stopped: "Stopped",
    valid_solver_ready: "Solver ready",
    valid_with_warnings: "Ready with warnings",
    need_clarification: "Needs information",
    unsupported_task: "Unsupported task",
    backend_error: "Backend error",
    rough_layout_generated: "Layout generated",
    evaluated: "Evaluated",
    not_evaluated: "Not evaluated",
    feasible: "Feasible",
    optimal: "Optimal",
    failed: "Failed",
  };
  if (exact[value]) return exact[value];
  if (value.includes("skip") || value.includes("not_invoked")) return "Not required";
  if (value.includes("clarification")) return "Needs information";
  if (value.includes("warning")) return "Completed with warnings";
  if (value.includes("fail") || value.includes("error") || value.includes("invalid")) return "Failed";
  if (value.includes("done") || value.includes("complete") || value.includes("valid")) return "Completed";
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function friendlyReason(reason) {
  return {
    quality_accepted: "Quality target accepted",
    intent_requires_clarification: "Waiting for missing information",
    user_clarification_required: "Waiting for missing information",
    unsupported_intent: "The task is not supported by the current layout pipeline",
    validation_complete: "Validation completed",
    evaluation_complete: "Evaluation completed",
    feasible_layout_generated: "Feasible layout generated",
  }[reason] || String(reason || "").replaceAll("_", " ");
}

function friendlyAgentStatus(status, rawStatus) {
  if (status === "running") return "Running";
  if (status === "completed") return "Completed";
  if (status === "warning") return "Completed with warnings";
  if (status === "failed") return "Failed";
  if (status === "skipped") return String(rawStatus || "").includes("not_evaluated") ? "Not evaluated" : "Not required";
  return "Pending";
}

function friendlyEventSummary(event) {
  const eventName = event.event || "message";
  const skipped = event.status === "skipped";
  const labels = {
    session_started: "Task started",
    rag_started: "Retrieving engineering evidence",
    rag_done: "Engineering RAG evidence attached",
    intent_started: "Understanding the request",
    intent_done: "Task type and requirements identified",
    design_started: "Designer started",
    design_done: skipped ? "Design and solver were not required" : "Layout candidate generated",
    validation_started: "Validation started",
    validation_done: skipped ? "Validation was not required" : "Constraint checks completed",
    evaluation_started: "Evaluation started",
    evaluation_done: skipped ? "Evaluation was not required" : "Layout evaluation completed",
    feedback_generated: "A revision was requested",
    loop_done: "Workflow route completed",
    viewport_command: "Layout sent to the 3D viewport",
    turn_done: "Task completed",
    turn_blocked: "Waiting for user input",
  };
  return labels[eventName] || "Workflow updated";
}

function activateContextTab(tab) {
  document.querySelector(`.context-tab[data-context-tab="${tab}"]`)?.click();
}

function bindTabs(buttonSelector, buttonDataKey, paneSelector, paneDataKey) {
  for (const button of document.querySelectorAll(buttonSelector)) {
    button.addEventListener("click", () => {
      const value = button.dataset[buttonDataKey];
      for (const candidate of document.querySelectorAll(buttonSelector)) candidate.classList.toggle("active", candidate === button);
      for (const pane of document.querySelectorAll(paneSelector)) pane.classList.toggle("active", pane.dataset[paneDataKey] === value);
    });
  }
}

function saveWorkspace() {
  const snapshot = buildWorkspaceSnapshot();
  localStorage.setItem("agentic-layopt-workspace", JSON.stringify(snapshot));
  showToast("Workspace saved in this browser.");
}

function restoreWorkspace() {
  const raw = localStorage.getItem("agentic-layopt-workspace");
  if (!raw) {
    showToast("No saved workspace found.");
    return;
  }
  try {
    const snapshot = JSON.parse(raw);
    el.prompt.value = snapshot.prompt || "";
    el.engineeringContext.value = snapshot.engineering_context || "";
    if (snapshot.solver && [...el.solverSelect.options].some((item) => item.value === snapshot.solver)) {
      el.solverSelect.value = snapshot.solver;
    }
    for (const [key, checked] of Object.entries(snapshot.objectives || {})) {
      if (el.objectives[key]) el.objectives[key].checked = Boolean(checked);
    }
    for (const [key, value] of Object.entries(snapshot.parameters || {})) {
      if (el.parameters[key]) el.parameters[key].value = value ?? "";
    }
    state.semanticOverrides = snapshot.semantic_overrides || {};
    renderSelectedSolver();
    updateObjectiveCount();
    showToast("Saved task restored.");
  } catch (_error) {
    showToast("Saved workspace could not be restored.");
  }
}

async function restoreCapturedWorkspaceFromUrl() {
  const restore = new URLSearchParams(window.location.search).get("restore");
  if (restore !== "captured") return;
  try {
    const response = await fetch("/api/workspace-snapshot", { cache: "no-store" });
    if (!response.ok) throw new Error(`Captured workspace HTTP ${response.status}`);
    const snapshot = await response.json();
    await applyCapturedWorkspace(snapshot);
  } catch (error) {
    showToast(`Captured workspace could not be loaded: ${error.message}`);
  }
}

async function applyCapturedWorkspace(snapshot) {
  if (!snapshot || typeof snapshot !== "object" || !snapshot.rough_layout) {
    throw new Error("The captured workspace does not contain a layout.");
  }
  state.capturedWorkspace = true;
  el.prompt.disabled = true;
  el.run.disabled = true;
  state.capturedElapsedText = snapshot.elapsed_text || null;
  state.capturedEquipmentNames = snapshot.equipment_display_names || [];
  state.sessionId = String(snapshot.session_id || state.sessionId);
  el.engineeringContext.value = snapshot.engineering_context || "";
  state.latestLayout = snapshot.rough_layout;
  state.pendingLayout = null;
  const capturedComparison = snapshot.change_comparison
    || snapshot.rough_layout.change_comparison
    || null;
  state.currentChangeComparison = capturedComparison?.before_layout || capturedComparison?.beforeLayout
    ? {
      stage: changeComparisonStage(capturedComparison.stage),
      beforeLayout: capturedComparison.before_layout || capturedComparison.beforeLayout,
    }
    : null;
  state.changeComparisonBeforeLayout = null;
  state.changeComparisonAwaitingClarification = false;
  state.latestResponse = snapshot.response || null;
  state.latestSolverReport = snapshot.solver_report || null;
  state.latestValidation = snapshot.validation_report || null;
  state.latestEvaluation = snapshot.evaluation || null;
  state.latestSemanticIntent = snapshot.semantic_intent || null;
  state.clarificationOptions = [];
  state.clarificationQuestions = [];
  state.semanticOverrides = {};
  state.selectedInstanceId = null;
  resetConversation();
  for (const stage of snapshot.stage_records || []) {
    if (stage.original_prompt) appendMessage("user", stage.original_prompt);
    for (const clarification of stage.clarification_attempts || []) {
      if (typeof clarification.reply === "string" && clarification.reply.trim()) {
        appendMessage("user", clarification.reply);
      }
    }
    if (stage.assistant_message) {
      appendMessage("assistant", stage.assistant_message, { rich: true });
    }
  }
  renderSemanticIntent(
    snapshot.semantic_intent,
    { intent: snapshot.semantic_intent?.intent_type },
  );
  renderValidation(snapshot.validation_report);
  renderEvaluation(snapshot.evaluation);
  renderSolverReport(snapshot.solver_report);
  const recorded = snapshot.agent_statuses || {};
  setAgentStates(
    recorded.intent || (snapshot.semantic_intent ? "completed" : "pending"),
    recorded.designer || (snapshot.flp_ir ? "completed" : "pending"),
    recorded.validator || snapshot.validation_report?.status,
    recorded.evaluator || snapshot.evaluation?.evaluation_status,
  );
  updateElapsedDisplay();
  renderClarification([]);
  renderRawArtifacts({
    flp_ir: snapshot.flp_ir,
    evaluation: snapshot.evaluation,
    validation_report: snapshot.validation_report,
  });
  setRequestStatus(snapshot.response?.status || "VALID_SOLVER_READY", "captured_actual_run");
  el.workflowStatus.textContent = `Recorded ${snapshot.recorded_stage || "session"} · read-only`;
  el.prompt.placeholder = "Recorded session (read-only).";
  el.engineeringContext.disabled = true;
  for (const control of el.requirements.panel.querySelectorAll("input, select, textarea, button")) {
    control.disabled = true;
  }
  await sendLayoutToViewport(snapshot.rough_layout, null, null, {
    changeComparison: state.currentChangeComparison,
  });
  state.latestViewportReceipt = snapshot.viewport_receipt || null;
  const rendered = snapshot.viewport_receipt?.rendered_instance_count
    || snapshot.rough_layout?.placements?.length
    || 0;
  el.layoutStatus.textContent = `${rendered} equipment loaded from a captured accepted run · local view`;
  updateSelection(null);
  updateDrawerStatus();
}

function exportWorkspace() {
  const snapshot = buildWorkspaceSnapshot();
  const blob = new Blob([JSON.stringify(snapshot, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `agentic-layopt-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
  showToast("Workspace exported.");
}

function buildWorkspaceSnapshot() {
  return {
    exported_at: new Date().toISOString(),
    session_id: state.sessionId,
    prompt: el.prompt.value,
    engineering_context: el.engineeringContext.value,
    change_comparison: state.currentChangeComparison,
    solver: el.solverSelect.value,
    objectives: Object.fromEntries(Object.entries(el.objectives).map(([key, node]) => [key, node.checked])),
    parameters: Object.fromEntries(Object.entries(el.parameters).map(([key, node]) => [key, node.value])),
    semantic_overrides: state.semanticOverrides,
    semantic_intent: state.latestSemanticIntent,
    solver_report: state.latestSolverReport,
    validation_report: state.latestValidation,
    evaluation: state.latestEvaluation,
    viewport_receipt: state.latestViewportReceipt,
    rough_layout: state.latestLayout,
    response: state.latestResponse,
  };
}

function showToast(message) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  el.toastRegion.append(toast);
  window.setTimeout(() => toast.remove(), 2600);
}

function setText(target, text) {
  target.textContent = text;
}

function setChip(target, text, tone) {
  target.textContent = text;
  target.className = `state-chip ${tone || "muted"}`;
}

function validationIssueCount() {
  return (state.latestValidation?.errors?.length || 0) + (state.latestValidation?.warnings?.length || 0);
}

function solverLabel(value) {
  if (value === "constraint_repair") return "Constraint repair";
  return SOLVER_LABELS[value] || value || "unknown";
}

function formatNumber(value) {
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 3 });
}

function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return formatNumber(value);
}

function optionalNumber(value) {
  if (value === "" || value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatDimensions(dimensions) {
  if (!dimensions || !Object.keys(dimensions).length) return "n/a";
  return `${formatMetric(dimensions.width_m)} × ${formatMetric(dimensions.height_m)} × ${formatMetric(dimensions.depth_m)} m`;
}

function formatFootprint(footprint) {
  if (!footprint || !Object.keys(footprint).length) return "n/a";
  return `x ${formatMetric(footprint.min_x)}…${formatMetric(footprint.max_x)}, z ${formatMetric(footprint.min_z)}…${formatMetric(footprint.max_z)}`;
}

function formatPosition(position) {
  if (!position) return "n/a";
  return `x ${formatMetric(position.x_m)}, y ${formatMetric(position.y_m)}, z ${formatMetric(position.z_m)} m`;
}

function formatObjectSummary(value) {
  if (value == null) return "Not provided";
  if (Array.isArray(value)) return value.length ? value.map((item) => typeof item === "string" ? item : JSON.stringify(item)).join(", ") : "None";
  if (typeof value === "object") return Object.keys(value).length ? JSON.stringify(value) : "None";
  return String(value);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
