import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";


export async function createLayoutViewport({ canvas, emptyState, manifestUrl, onSelect }) {
  const response = await fetch(manifestUrl, { cache: "no-cache" });
  if (!response.ok) throw new Error(`Model manifest HTTP ${response.status}`);
  const manifest = await response.json();
  const assets = new Map(
    (manifest.assets || []).map((item) => [item.model_asset_id, item]),
  );
  if (!assets.size) throw new Error("Model manifest contains no assets");
  return new LayoutViewport(canvas, emptyState, assets, onSelect);
}


class LayoutViewport {
  constructor(canvas, emptyState, assets, onSelect) {
    this.canvas = canvas;
    this.emptyState = emptyState;
    this.assets = assets;
    this.onSelect = onSelect;
    this.modelCache = new Map();
    this.fixedGeometryPrototype = fixedGeometryPrototype();
    this.loadedModelIds = new Set();
    this.instanceRoots = new Map();
    this.renderVersion = 0;
    this.selectedInstanceId = null;
    this.flags = {
      showBoundary: true,
      showOverlap: true,
      showFlow: true,
      showGrid: true,
      showModels: true,
      showWalls: true,
      showAisles: true,
      showMaintenance: true,
      showRegions: true,
      showChanges: true,
    };
    this.pointerDown = new THREE.Vector2();
    this.lastSite = { width: 20, depth: 12 };
    this.lastCameraBounds = null;
    this.lastCameraFrame = null;
    this.cameraMode = "perspective";
    this.currentLayout = null;
    this.changeComparison = null;
    this.changeSummary = null;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x3f4a55);
    this.perspectiveCamera = new THREE.PerspectiveCamera(46, 1, 0.05, 2000);
    this.orthographicCamera = new THREE.OrthographicCamera(-10, 10, 10, -10, 0.05, 2000);
    this.camera = this.perspectiveCamera;
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      powerPreference: "high-performance",
    });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = false;
    this.controls.screenSpacePanning = true;
    this.controls.minPolarAngle = THREE.MathUtils.degToRad(12);
    this.controls.maxPolarAngle = THREE.MathUtils.degToRad(86);
    this.controls.addEventListener("change", () => this.draw());

    this.loader = new GLTFLoader();
    this.loader.setMeshoptDecoder(MeshoptDecoder);
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();

    this.addLights();
    this.resetGroups();
    this.bindInteraction();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement);
    this.resize();
    this.frameCamera({ width: 20, depth: 12 });
    this.draw();
  }

  addLights() {
    const hemisphere = new THREE.HemisphereLight(0xe9f3ff, 0x26313a, 2.65);
    this.scene.add(hemisphere);

    const key = new THREE.DirectionalLight(0xffffff, 3.2);
    key.position.set(-18, 28, -16);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 0.5;
    key.shadow.camera.far = 150;
    key.shadow.bias = -0.00025;
    this.scene.add(key);

    const fill = new THREE.DirectionalLight(0x8fc4ff, 1.15);
    fill.position.set(20, 12, 18);
    this.scene.add(fill);

    const rim = new THREE.DirectionalLight(0xd5e5f4, 0.65);
    rim.position.set(-4, 10, 30);
    this.scene.add(rim);
  }

  resetGroups() {
    if (this.selectionHelper) {
      this.scene.remove(this.selectionHelper);
      this.selectionHelper.geometry.dispose();
      this.selectionHelper.material.dispose();
      this.selectionHelper = null;
    }
    if (this.layoutRoot) {
      disposeObject(this.siteGroup);
      disposeObject(this.gridGroup);
      disposeObject(this.wallGroup);
      disposeObject(this.aisleGroup);
      disposeObject(this.maintenanceGroup);
      disposeObject(this.boundaryGroup);
      disposeObject(this.flowGroup);
      disposeObject(this.diagnosticGroup);
      disposeObject(this.regionGroup);
      disposeObject(this.changeOverlayGroup);
      this.scene.remove(this.layoutRoot);
    }
    this.layoutRoot = new THREE.Group();
    this.siteGroup = new THREE.Group();
    this.gridGroup = new THREE.Group();
    this.wallGroup = new THREE.Group();
    this.aisleGroup = new THREE.Group();
    this.maintenanceGroup = new THREE.Group();
    this.boundaryGroup = new THREE.Group();
    this.equipmentGroup = new THREE.Group();
    this.flowGroup = new THREE.Group();
    this.diagnosticGroup = new THREE.Group();
    this.regionGroup = new THREE.Group();
    this.changeOverlayGroup = new THREE.Group();
    this.layoutRoot.add(
      this.siteGroup,
      this.gridGroup,
      this.wallGroup,
      this.aisleGroup,
      this.maintenanceGroup,
      this.boundaryGroup,
      this.equipmentGroup,
      this.flowGroup,
      this.diagnosticGroup,
      this.regionGroup,
      this.changeOverlayGroup,
    );
    this.scene.add(this.layoutRoot);
    this.applyFlags();
  }

  async render(layout, receiptContext = {}) {
    const version = ++this.renderVersion;
    const layoutSnapshot = cloneJson(receiptContext.layoutSnapshot || layout || {});
    const receiptIdentity = {
      turn_id: receiptContext.turnId ?? null,
      layout_sha256: receiptContext.layoutSha256 || null,
      layout_snapshot: layoutSnapshot,
      render_token: receiptContext.renderToken || null,
    };
    const expectedInstanceIds = expectedViewportInstanceIds(layout);
    this.currentLayout = layout || null;
    this.changeComparison = null;
    this.changeSummary = null;
    this.selectedInstanceId = null;
    this.instanceRoots.clear();
    this.resetGroups();
    if (!layout?.site) {
      this.setEmpty("Run a request to render a layout.", true);
      const telemetry = await this.captureFrame();
      return {
        schema_version: "3.0",
        renderer: "threejs",
        render_status: "EMPTY",
        layout_id: layout?.layout_id || null,
        render_version: version,
        rendered_instance_ids: [],
        expected_instance_ids: expectedInstanceIds,
        missing_instance_ids: expectedInstanceIds,
        failed_instance_ids: [],
        failed_model_asset_ids: [],
        asset_load_failures: [],
        viewport_visibility: emptyViewportVisibility(expectedInstanceIds),
        instance_records: [],
        model_load_reports: [],
        browser: browserMetadata(),
        webgl: telemetry.webgl,
        frame: telemetry.frame,
        ...receiptIdentity,
      };
    }

    const site = siteDimensions(layout.site);
    this.renderSite(layout.site, site);
    this.frameCamera(site);
    const placements = [
      ...(layout.placements || []).map((item) => ({ ...item, fixed: false })),
      ...fixedEquipmentPlacements(layout.site),
    ];
    const uniqueAssetIds = [...new Set(
      placements.flatMap(placementModelAssetIds).filter(Boolean),
    )];
    this.setEmpty(`Loading models 0/${uniqueAssetIds.length}`, false);

    let completed = 0;
    const models = new Map();
    const failedModelAssetIds = [];
    const modelLoadReports = [];
    await Promise.all(uniqueAssetIds.map(async (modelAssetId) => {
      const report = { model_asset_id: modelAssetId, attempts: 0, cache_hit: false };
      try {
        models.set(modelAssetId, await this.loadModel(modelAssetId, report));
        report.status = "LOADED";
      } catch (error) {
        console.error(`Failed to load ${modelAssetId}`, error);
        models.set(modelAssetId, null);
        failedModelAssetIds.push(modelAssetId);
        report.status = "FAILED";
        report.error = String(error?.message || error);
      } finally {
        modelLoadReports.push(report);
        completed += 1;
        if (version === this.renderVersion) {
          this.setEmpty(`Loading models ${completed}/${uniqueAssetIds.length}`, false);
        }
      }
    }));
    if (version !== this.renderVersion) {
      return {
        schema_version: "3.0",
        renderer: "threejs",
        render_status: "SUPERSEDED",
        layout_id: layout.layout_id || null,
        render_version: version,
        rendered_instance_ids: [],
        expected_instance_ids: expectedInstanceIds,
        missing_instance_ids: expectedInstanceIds,
        failed_instance_ids: [],
        failed_model_asset_ids: [...new Set(failedModelAssetIds)].sort(),
        asset_load_failures: failedAssetReports(modelLoadReports),
        viewport_visibility: emptyViewportVisibility(expectedInstanceIds),
        instance_records: [],
        model_load_reports: modelLoadReports.sort(compareModelReports),
        browser: browserMetadata(),
        webgl: webglMetadata(this.renderer.getContext()),
        frame: null,
        ...receiptIdentity,
      };
    }

    const renderedInstanceIds = [];
    const failedInstanceIds = [];
    const instanceRecords = [];
    const renderedPlacements = [];
    for (const placement of placements) {
      const compound = Array.isArray(placement.compound_members)
        && placement.compound_members.length > 0;
      const renderSource = compound
        ? "compound_model"
        : placement.model_asset_id ? "model" : "geometry_placeholder";
      const prototype = compound
        ? this.fixedGeometryPrototype
        : placement.model_asset_id
          ? models.get(placement.model_asset_id)
          : placement.fixed ? this.fixedGeometryPrototype : null;
      if (!prototype) {
        if (placement.instance_id) failedInstanceIds.push(placement.instance_id);
        continue;
      }
      try {
        const wrapper = this.renderPlacement(placement, prototype, models);
        renderedPlacements.push({ placement, wrapper, renderSource });
        if (placement.instance_id) renderedInstanceIds.push(placement.instance_id);
      } catch (error) {
        console.error(`Failed to render ${placement.instance_id || "instance"}`, error);
        if (placement.instance_id) failedInstanceIds.push(placement.instance_id);
      }
    }
    this.renderOverlapDiagnostics(placements);
    this.renderFlow(placements, layout.flow_relations || [], layout.site?.site_features || []);
    const comparison = receiptContext.changeComparison || layout.change_comparison || null;
    const beforeLayout = comparison?.beforeLayout || comparison?.before_layout || null;
    const comparisonStage = comparison?.stage || receiptContext.stage || layout.stage || "";
    if (beforeLayout) this.setChangeComparison(beforeLayout, comparisonStage, false);
    this.frameCamera(site, sceneBoundsForCamera(this.layoutRoot, site));
    for (const rendered of renderedPlacements) {
      const record = instanceRenderRecord(
        rendered.placement,
        rendered.wrapper,
        rendered.renderSource,
        this.camera,
        this.canvas,
      );
      instanceRecords.push(record);
      if (!record.finite_world_bounds || record.mesh_count <= 0) {
        failedInstanceIds.push(rendered.placement.instance_id);
      }
    }
    this.setEmpty("", true);
    this.applyFlags();
    const telemetry = await this.captureFrame();
    if (version !== this.renderVersion) {
      return {
        schema_version: "3.0",
        renderer: "threejs",
        render_status: "SUPERSEDED",
        layout_id: layout.layout_id || null,
        render_version: version,
        rendered_instance_ids: [],
        expected_instance_ids: expectedInstanceIds,
        missing_instance_ids: expectedInstanceIds,
        failed_instance_ids: [],
        failed_model_asset_ids: [...new Set(failedModelAssetIds)].sort(),
        asset_load_failures: failedAssetReports(modelLoadReports),
        viewport_visibility: emptyViewportVisibility(expectedInstanceIds),
        instance_records: [],
        model_load_reports: modelLoadReports.sort(compareModelReports),
        browser: browserMetadata(),
        webgl: telemetry.webgl,
        frame: telemetry.frame,
        ...receiptIdentity,
      };
    }
    const telemetryFailed = telemetry.webgl.context_lost
      || Number(telemetry.webgl.gl_error || 0) !== 0
      || !telemetry.frame.pixel_sha256
      || !telemetry.frame.screenshot_sha256;
    const renderedUnique = [...new Set(renderedInstanceIds)].sort();
    const missingInstanceIds = expectedInstanceIds.filter(
      (instanceId) => !renderedUnique.includes(instanceId),
    );
    const visibility = viewportVisibility(instanceRecords, expectedInstanceIds);
    const emptyFrame = expectedInstanceIds.length > 0
      && (
        renderedUnique.length === 0
        || visibility.intersecting_instance_ids.length === 0
        || Number(telemetry.frame.distinct_rgb_sample_count || 0) < 2
      );
    return {
      schema_version: "3.0",
      renderer: "threejs",
      render_status: failedInstanceIds.length
        || failedModelAssetIds.length
        || missingInstanceIds.length
        || visibility.non_intersecting_instance_ids.length
        || telemetryFailed
        || emptyFrame
        ? "PARTIAL"
        : "RENDERED",
      layout_id: layout.layout_id || null,
      render_version: version,
      expected_instance_ids: expectedInstanceIds,
      rendered_instance_ids: renderedUnique,
      missing_instance_ids: missingInstanceIds,
      failed_instance_ids: [...new Set(failedInstanceIds)].sort(),
      failed_model_asset_ids: [...new Set(failedModelAssetIds)].sort(),
      asset_load_failures: failedAssetReports(modelLoadReports),
      instance_records: instanceRecords.sort(compareInstanceRecords),
      viewport_visibility: visibility,
      empty_frame: emptyFrame,
      model_load_reports: modelLoadReports.sort(compareModelReports),
      browser: browserMetadata(),
      webgl: telemetry.webgl,
      frame: telemetry.frame,
      ...receiptIdentity,
    };
  }

  clear() {
    this.renderVersion += 1;
    this.selectedInstanceId = null;
    this.currentLayout = null;
    this.changeComparison = null;
    this.changeSummary = null;
    this.instanceRoots.clear();
    this.resetGroups();
    this.setEmpty("Run a request to render a layout.", false);
    this.frameCamera({ width: 20, depth: 12 });
    this.onSelect?.(null);
    this.draw();
  }

  setFlags(flags) {
    this.flags = { ...this.flags, ...flags };
    this.applyFlags();
    this.draw();
  }

  applyFlags() {
    if (!this.boundaryGroup) return;
    this.boundaryGroup.visible = this.flags.showBoundary;
    this.flowGroup.visible = this.flags.showFlow;
    this.diagnosticGroup.visible = this.flags.showOverlap;
    this.gridGroup.visible = this.flags.showGrid;
    this.equipmentGroup.visible = this.flags.showModels;
    this.wallGroup.visible = this.flags.showWalls;
    this.aisleGroup.visible = this.flags.showAisles;
    this.maintenanceGroup.visible = this.flags.showMaintenance;
    this.regionGroup.visible = this.flags.showRegions !== false;
    this.changeOverlayGroup.visible = this.flags.showChanges !== false;
  }

  setChangeComparison(beforeLayout, stage = "edit", redraw = true) {
    const stageName = String(stage || "").toLowerCase();
    if (!this.currentLayout || !beforeLayout || !["edit", "repair"].includes(stageName)) {
      this.changeComparison = null;
      this.changeSummary = null;
      this.clearChangeOverlay();
      if (redraw) this.draw();
      return null;
    }
    this.changeComparison = { beforeLayout, stage: stageName };
    this.clearChangeOverlay();
    const changes = compareLayoutChanges(beforeLayout, this.currentLayout);
    this.changeSummary = renderLayoutChangeOverlays(this.changeOverlayGroup, changes);
    this.changeOverlayGroup.visible = this.flags.showChanges !== false;
    if (redraw) this.draw();
    return this.changeSummary;
  }

  clearChangeOverlay() {
    if (!this.changeOverlayGroup) return;
    for (const child of [...this.changeOverlayGroup.children]) {
      this.changeOverlayGroup.remove(child);
      disposeObject(child);
    }
  }

  async loadModel(modelAssetId, report = {}) {
    if (this.modelCache.has(modelAssetId)) {
      try {
        const cached = await this.modelCache.get(modelAssetId);
        report.cache_hit = true;
        return cached;
      } catch (_error) {
        this.modelCache.delete(modelAssetId);
      }
    }
    const asset = this.assets.get(modelAssetId);
    if (!asset?.glb_url) throw new Error("No GLB mapping in runtime manifest");
    let lastError = null;
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      report.attempts = attempt;
      const promise = this.loader.loadAsync(asset.glb_url).then((gltf) => {
        this.loadedModelIds.add(modelAssetId);
        return prepareModelAppearance(gltf.scene);
      });
      this.modelCache.set(modelAssetId, promise);
      try {
        return await promise;
      } catch (error) {
        lastError = error;
        this.modelCache.delete(modelAssetId);
        if (attempt < 3) await delay(100 * attempt);
      }
    }
    throw lastError || new Error(`Failed to load ${modelAssetId}`);
  }

  renderSite(siteData, site) {
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(site.width, site.depth),
      new THREE.MeshStandardMaterial({ color: 0x35424d, roughness: 0.88, metalness: 0.04 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.015;
    floor.receiveShadow = true;
    this.siteGroup.add(floor);

    const grid = createSiteGrid(site);
    grid.position.y = 0.008;
    this.gridGroup.add(grid);

    for (const zone of siteData.non_placeable_zones || []) {
      const width = Number(zone.width_m || Number(zone.max_x_m) - Number(zone.min_x_m));
      const depth = Number(zone.depth_m || Number(zone.max_z_m) - Number(zone.min_z_m));
      if (!(width > 0 && depth > 0)) continue;
      const zoneType = String(zone.zone_type || "restricted").toLowerCase();
      const wall = zoneType === "wall";
      const maintenance = zoneType.includes("maintenance") || zoneType.includes("clearance");
      const aisle = zoneType.includes("aisle") || zoneType.includes("corridor");
      const height = wall ? 0.72 : 0.045;
      const material = new THREE.MeshStandardMaterial({
        color: wall ? 0x1e272e : maintenance ? 0x6f9b7a : aisle ? 0xa8afb4 : 0x8d7f8f,
        transparent: !wall,
        opacity: wall ? 1 : 0.72,
        roughness: wall ? 0.72 : 0.95,
        metalness: wall ? 0.12 : 0,
      });
      const area = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), material);
      area.position.set(
        (Number(zone.min_x_m) + Number(zone.max_x_m)) * 0.5,
        height * 0.5,
        (Number(zone.min_z_m) + Number(zone.max_z_m)) * 0.5,
      );
      area.receiveShadow = true;
      if (wall) this.wallGroup.add(area);
      else if (maintenance) this.maintenanceGroup.add(area);
      else this.aisleGroup.add(area);
    }

    for (const feature of siteData.site_features || []) {
      if (!/buffer/i.test(String(feature.feature_name || "")) || !feature.pose || !feature.dimensions_m) continue;
      const { width_m: width, depth_m: depth } = feature.dimensions_m;
      const x = Number(feature.pose.x_m), z = Number(feature.pose.z_m);
      const points = [];
      for (let offset = -depth; offset <= width; offset += 0.4) {
        const lo = Math.max(0, -offset), hi = Math.min(depth, width - offset);
        if (hi <= lo) continue;
        points.push(new THREE.Vector3(x - width / 2 + lo + offset, 0.09, z - depth / 2 + lo),
          new THREE.Vector3(x - width / 2 + hi + offset, 0.09, z - depth / 2 + hi));
      }
      this.aisleGroup.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points),
        new THREE.LineBasicMaterial({ color: 0xe3bc5b })));
      const label = floorText(String(feature.feature_name).replace(/\b\w/g, (c) => c.toUpperCase()), Math.min(width * 0.9, 3));
      label.position.set(x, 0.10, z); this.aisleGroup.add(label);
    }
    for (const region of siteData.placement_regions || []) {
      const id = String(region.zone_id || "");
      const label = id === "functional/northwest" ? "ASSEMBLY & TRANSFER"
        : id === "functional/northeast" ? "TEST & FINISHING"
          : /placement_south/.test(id) ? "MACHINING" : null;
      if (!label) continue;
      const text = floorText(label, 8);
      text.position.set((Number(region.min_x_m) + Number(region.max_x_m)) / 2, 0.07, Number(region.max_z_m) - 0.7);
      this.aisleGroup.add(text);
    }
    for (const cell of siteData.workcell_areas || []) {
      const area = cell.operator_area;
      if (!area) continue;
      const patch = new THREE.Mesh(new THREE.PlaneGeometry(area.max_x - area.min_x, area.max_z - area.min_z),
        new THREE.MeshStandardMaterial({ color: 0x6c8278, roughness: 0.95, transparent: true, opacity: 0.5 }));
      patch.rotation.x = -Math.PI / 2;
      patch.position.set((area.min_x + area.max_x) / 2, 0.075, (area.min_z + area.max_z) / 2);
      this.aisleGroup.add(patch);
    }
    this.renderSiteRegionAnnotations(siteData);
    this.renderPerimeterDoors(siteData, site);
    const points = [
      new THREE.Vector3(-site.width / 2, 0.08, -site.depth / 2),
      new THREE.Vector3(site.width / 2, 0.08, -site.depth / 2),
      new THREE.Vector3(site.width / 2, 0.08, site.depth / 2),
      new THREE.Vector3(-site.width / 2, 0.08, site.depth / 2),
    ];
    const boundary = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({ color: 0x76b9ef }),
    );
    this.boundaryGroup.add(boundary);
  }

  renderSiteRegionAnnotations(siteData) {
    const annotations = siteAnnotationRecords(siteData);
    for (const region of annotations.regions) {
      addFloorRegion(this.regionGroup, region, {
        fill: 0x91a69a,
        fillOpacity: 0.11,
        edge: 0xb7c6bb,
        edgeOpacity: 0.72,
        labelColor: "#e0e7e2",
      });
    }
    for (const bay of annotations.bays) {
      addFloorRegion(this.regionGroup, bay, bay.reserved
        ? { fill: 0xb7a36d, fillOpacity: 0.10, edge: 0xc2b17f, edgeOpacity: 0.78, labelColor: "#e2d2a6" }
        : { fill: 0x9baea2, fillOpacity: 0.025, edge: 0x9ca9a2, edgeOpacity: 0.5, edgeOnly: true });
      if (bay.reserved) addFloorHatch(this.regionGroup, bay.bounds, 0xd1b875);
    }
  }

  renderPerimeterDoors(siteData, site) {
    const doors = (siteData.site_features || []).filter((item) => ["entrance", "exit"].includes(item.feature_type));
    if (!doors.length || (siteData.non_placeable_zones || []).some((zone) => zone.zone_type === "wall")) return;
    // Wall height and thickness are display settings; the bounded floor is the usable interior.
    const height = 4.2, thickness = 0.2;
    const wall = new THREE.MeshStandardMaterial({ color: 0x22272a, roughness: 0.76, metalness: 0.12 });
    const frame = new THREE.MeshStandardMaterial({ color: 0x78838a, roughness: 0.45, metalness: 0.6 });
    const box = (x, y, z, w, h, d, material) => {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
      mesh.position.set(x, y, z); mesh.castShadow = true; mesh.receiveShadow = true;
      this.wallGroup.add(mesh);
    };
    for (const side of [
      { axis: "x", coordinate: -site.width / 2, span: site.depth },
      { axis: "x", coordinate: site.width / 2, span: site.depth },
      { axis: "z", coordinate: -site.depth / 2, span: site.width },
      { axis: "z", coordinate: site.depth / 2, span: site.width },
    ]) {
      const along = side.axis === "x" ? "z_m" : "x_m";
      const across = side.axis === "x" ? "x_m" : "z_m";
      const sizeKey = side.axis === "x" ? "depth_m" : "width_m";
      const openings = doors.filter((door) => Math.abs(Number(door.pose?.[across]) - side.coordinate) < 1e-3)
        .map((door) => ({ center: Number(door.pose?.[along] || 0), width: Number(door.dimensions_m?.[sizeKey] || 0) }))
        .filter((opening) => opening.width > 0).sort((a, b) => a.center - b.center);
      const fixed = side.coordinate + Math.sign(side.coordinate) * thickness / 2;
      const segment = (lo, hi, y, h, material = wall, wallThickness = thickness) => {
        if (hi <= lo) return;
        if (side.axis === "x") box(fixed, y, (lo + hi) / 2, wallThickness, h, hi - lo, material);
        else box((lo + hi) / 2, y, fixed, hi - lo, h, wallThickness, material);
      };
      let cursor = -side.span / 2;
      for (const opening of openings) {
        const lo = Math.max(-side.span / 2, opening.center - opening.width / 2);
        const hi = Math.min(side.span / 2, opening.center + opening.width / 2);
        segment(cursor, lo, height / 2, height);
        segment(lo, hi, 3.9, 0.6);
        segment(lo - 0.12, lo, 1.8, 3.6, frame, 0.28);
        segment(hi, hi + 0.12, 1.8, 3.6, frame, 0.28);
        cursor = Math.max(cursor, hi);
      }
      segment(cursor, side.span / 2, height / 2, height);
    }
  }

  renderPlacement(placement, prototype, models = new Map()) {
    const dimensions = placement.dimensions_m || {};
    const wrapper = new THREE.Group();
    wrapper.name = placement.instance_id || "equipment";
    wrapper.userData.placement = placement;
    wrapper.rotation.y = -THREE.MathUtils.degToRad(Number(placement.rotation?.yaw_deg || 0));
    wrapper.position.set(
      Number(placement.position?.x_m || 0),
      0.02,
      Number(placement.position?.z_m || 0),
    );

    if (Array.isArray(placement.compound_members) && placement.compound_members.length) {
      for (const member of placement.compound_members) {
        const memberPrototype = models.get(member.model_asset_id);
        if (!memberPrototype) continue;
        const memberPose = member.relative_pose || {};
        const memberContainer = new THREE.Group();
        memberContainer.name = member.member_id || "compound-member";
        memberContainer.position.set(
          Number(memberPose.x_m || 0),
          0,
          Number(memberPose.z_m || 0),
        );
        memberContainer.rotation.y = -THREE.MathUtils.degToRad(
          Number(memberPose.yaw_deg || 0),
        );
        memberContainer.add(
          scaledModel(
            memberPrototype,
            member.dimensions_m || {},
            wrapper,
          ),
        );
        wrapper.add(memberContainer);
      }
    } else {
      wrapper.add(scaledModel(prototype, dimensions, wrapper, Number(placement.rotation?.yaw_deg || 0)));
    }
    this.equipmentGroup.add(wrapper);
    if (placement.instance_id) this.instanceRoots.set(placement.instance_id, wrapper);
    return wrapper;
  }

  renderFlow(placements, flowRelations, siteFeatures = []) {
    const byId = new Map(placements.map((item) => [item.instance_id, item]));
    for (const feature of siteFeatures) {
      if (feature.site_feature_id && feature.pose) {
        byId.set(feature.site_feature_id, { position: feature.pose, dimensions_m: feature.dimensions_m });
      }
    }
    const reviewedRelations = (flowRelations || []).filter(isReviewedFlowRelation);
    for (const relation of reviewedRelations) {
      const fromPlacement = byId.get(relation.from_instance_id);
      const toPlacement = byId.get(relation.to_instance_id);
      if (!fromPlacement || !toPlacement) continue;
      const height = Math.max(
        Number(fromPlacement.dimensions_m?.height_m || 1),
        Number(toPlacement.dimensions_m?.height_m || 1),
      ) + 0.25;
      const from = placementCenter(fromPlacement, height);
      const to = placementCenter(toPlacement, height);
      const direction = to.clone().sub(from);
      const length = direction.length();
      if (length < 1e-4) continue;
      const arrow = new THREE.ArrowHelper(direction.normalize(), from, length, 0xe4b454, 0.34, 0.16);
      this.flowGroup.add(arrow);
    }
  }

  async captureFrame() {
    await nextFrame(() => this.draw());
    const gl = this.renderer.getContext();
    gl.finish?.();
    const width = Number(this.canvas.width || 0);
    const height = Number(this.canvas.height || 0);
    let pixelSha256 = null;
    let screenshotSha256 = null;
    let nonzeroRgbPixelCount = 0;
    const sampledColors = new Set();
    if (width > 0 && height > 0 && !gl.isContextLost()) {
      const pixels = new Uint8Array(width * height * 4);
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      for (let index = 0; index < pixels.length; index += 4) {
        if (pixels[index] || pixels[index + 1] || pixels[index + 2]) {
          nonzeroRgbPixelCount += 1;
        }
        if (index % Math.max(4, Math.floor(pixels.length / 4096 / 4) * 4) === 0) {
          sampledColors.add(
            `${pixels[index]}:${pixels[index + 1]}:${pixels[index + 2]}`,
          );
        }
      }
      pixelSha256 = await sha256Hex(pixels);
      screenshotSha256 = await canvasPngSha256(this.canvas);
    }
    const webgl = webglMetadata(gl);
    return {
      webgl,
      frame: {
        width_px: width,
        height_px: height,
        pixel_sha256: pixelSha256,
        screenshot_sha256: screenshotSha256,
        nonzero_rgb_pixel_count: nonzeroRgbPixelCount,
        distinct_rgb_sample_count: sampledColors.size,
      },
    };
  }

  renderOverlapDiagnostics(placements) {
    for (const placement of placements) {
      if (!overlapsAny(placement, placements)) continue;
      const dimensions = placement.dimensions_m || {};
      const geometry = new THREE.BoxGeometry(
        Number(dimensions.width_m || 1),
        Number(dimensions.height_m || 1),
        Number(dimensions.depth_m || 1),
      );
      const outline = new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry),
        new THREE.LineBasicMaterial({ color: 0xf06b6b }),
      );
      outline.position.set(
        Number(placement.position?.x_m || 0),
        Number(dimensions.height_m || 1) * 0.5,
        Number(placement.position?.z_m || 0),
      );
      outline.rotation.y = -THREE.MathUtils.degToRad(Number(placement.rotation?.yaw_deg || 0));
      this.diagnosticGroup.add(outline);
    }
  }

  frameCamera(site, sceneBounds = null) {
    this.lastSite = site;
    const bounds = normalizedCameraBounds(sceneBounds, site);
    const target = bounds.getCenter(new THREE.Vector3());
    const radius = Math.max(4, bounds.getSize(new THREE.Vector3()).length() * 0.5);
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const aspect = Math.max(0.1, rect.width / Math.max(1, rect.height));
    const verticalFov = THREE.MathUtils.degToRad(this.perspectiveCamera.fov);
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov * 0.5) * aspect);
    const limitingFov = Math.max(
      THREE.MathUtils.degToRad(5),
      Math.min(verticalFov, horizontalFov),
    );
    const distance = radius / Math.sin(limitingFov * 0.5) * 1.14;
    const span = Math.max(site.width, site.depth, radius * 2, 8);
    const viewDirection = new THREE.Vector3(0.62, 0.72, 0.78).normalize();
    for (const camera of [this.perspectiveCamera, this.orthographicCamera]) {
      camera.near = Math.max(0.05, Math.min(radius * 0.02, distance * 0.05));
      camera.far = Math.max(span * 12, distance + radius * 4);
      camera.position.copy(target).addScaledVector(viewDirection, distance);
      camera.lookAt(target);
      camera.updateProjectionMatrix();
    }
    this.lastCameraBounds = bounds.clone();
    this.lastCameraFrame = { target: target.clone(), radius, distance };
    this.controls.target.copy(target);
    this.controls.minDistance = Math.max(radius * 0.4, span * 0.12);
    this.controls.maxDistance = Math.max(distance * 3, radius * 5);
    this.controls.update();
    this.updateOrthographicFrustum();
  }

  setCameraMode(mode) {
    const nextMode = mode === "orthographic" ? "orthographic" : "perspective";
    if (nextMode === this.cameraMode) return;
    const previous = this.camera;
    this.cameraMode = nextMode;
    this.camera = nextMode === "orthographic" ? this.orthographicCamera : this.perspectiveCamera;
    this.camera.position.copy(previous.position);
    this.camera.quaternion.copy(previous.quaternion);
    this.controls.object = this.camera;
    this.updateOrthographicFrustum();
    this.controls.update();
    this.draw();
  }

  resetCamera() {
    this.frameCamera(this.lastSite, this.lastCameraBounds);
    this.draw();
  }

  updateOrthographicFrustum() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const aspect = Math.max(0.1, rect.width / Math.max(1, rect.height));
    const radius = this.lastCameraFrame?.radius
      || Math.max(this.lastSite.width, this.lastSite.depth, 8) * 0.5;
    const span = Math.max(4, radius * 1.14 / Math.min(1, aspect));
    this.orthographicCamera.left = -span * aspect;
    this.orthographicCamera.right = span * aspect;
    this.orthographicCamera.top = span;
    this.orthographicCamera.bottom = -span;
    this.orthographicCamera.updateProjectionMatrix();
  }

  bindInteraction() {
    this.canvas.addEventListener("pointerdown", (event) => {
      this.pointerDown.set(event.clientX, event.clientY);
    });
    this.canvas.addEventListener("pointerup", (event) => {
      if (this.pointerDown.distanceTo(new THREE.Vector2(event.clientX, event.clientY)) > 5) return;
      this.selectAt(event.clientX, event.clientY);
    });
  }

  selectAt(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.set(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.equipmentGroup.children, true);
    const root = hits.map((hit) => hit.object.userData.instanceRoot).find(Boolean) || null;
    this.setSelection(root);
  }

  setSelection(root) {
    if (this.selectionHelper) {
      this.scene.remove(this.selectionHelper);
      this.selectionHelper.geometry.dispose();
      this.selectionHelper.material.dispose();
      this.selectionHelper = null;
    }
    this.selectedInstanceId = root?.userData?.placement?.instance_id || null;
    if (root) {
      this.selectionHelper = new THREE.BoxHelper(root, 0x64dfaf);
      this.scene.add(this.selectionHelper);
    }
    this.onSelect?.(root?.userData?.placement || null);
    this.draw();
  }

  highlightViolations(report) {
    this.flags.showOverlap = true;
    this.applyFlags();
    const serialized = JSON.stringify(report || {});
    const targetId = [...this.instanceRoots.keys()].find((instanceId) => serialized.includes(instanceId));
    if (targetId) this.setSelection(this.instanceRoots.get(targetId));
    else this.draw();
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(width, height, false);
    this.perspectiveCamera.aspect = width / height;
    this.perspectiveCamera.updateProjectionMatrix();
    this.frameCamera(this.lastSite, this.lastCameraBounds);
    this.draw();
  }

  setEmpty(text, hidden) {
    if (!this.emptyState) return;
    this.emptyState.textContent = text;
    this.emptyState.hidden = hidden;
  }

  draw() {
    this.renderer.render(this.scene, this.camera);
  }

  debugState() {
    return {
      renderer: "threejs",
      manifestAssetCount: this.assets.size,
      cachedAssetCount: this.modelCache.size,
      loadedAssetCount: this.loadedModelIds.size,
      renderedInstanceCount: this.equipmentGroup?.children.length || 0,
      siteObjectCount: this.siteGroup?.children.length || 0,
      selectedInstanceId: this.selectedInstanceId,
      cameraMode: this.cameraMode,
      showRegions: this.flags.showRegions !== false,
      showChanges: this.flags.showChanges !== false,
      changeSummary: this.changeSummary,
    };
  }
}


function placementModelAssetIds(placement) {
  if (Array.isArray(placement?.compound_members) && placement.compound_members.length) {
    return placement.compound_members.map((member) => member?.model_asset_id);
  }
  return [placement?.model_asset_id];
}


function scaledModel(prototype, dimensions, instanceRoot, footprintYawDeg = 0) {
  const target = new THREE.Vector3(
    Math.max(0.05, Number(dimensions?.width_m || 1)),
    Math.max(0.05, Number(dimensions?.height_m || 1)),
    Math.max(0.05, Number(dimensions?.depth_m || 1)),
  );
  // Placement dimensions describe the oriented world envelope. Restore local
  // dimensions before rotating the wrapper; otherwise a quarter-turn swaps
  // width/depth twice and distorts the displayed machine.
  const yaw = ((footprintYawDeg % 360) + 360) % 360;
  if (Math.abs((yaw % 180) - 90) < 1e-6) {
    [target.x, target.z] = [target.z, target.x];
  }
  const model = prototype.clone(true);
  model.updateMatrixWorld(true);
  const initialBounds = new THREE.Box3().setFromObject(model);
  const size = initialBounds.getSize(new THREE.Vector3());
  // The solver's footprint comes from the same catalog record as the runtime
  // model. Scale each axis to that declared geometry so the rendered object,
  // aisle, zones, and validation footprint stay in the same coordinate frame.
  model.scale.multiply(new THREE.Vector3(
    target.x / Math.max(size.x, 1e-6),
    target.y / Math.max(size.y, 1e-6),
    target.z / Math.max(size.z, 1e-6),
  ));
  model.updateMatrixWorld(true);
  const scaledBounds = new THREE.Box3().setFromObject(model);
  const center = scaledBounds.getCenter(new THREE.Vector3());
  model.position.x -= center.x;
  model.position.z -= center.z;
  model.position.y -= scaledBounds.min.y;
  model.traverse((child) => {
    if (!child.isMesh) return;
    child.castShadow = true;
    child.receiveShadow = true;
    child.userData.instanceRoot = instanceRoot;
  });
  return model;
}


function prepareModelAppearance(root) {
  let meshCount = 0;
  const lineObjects = [];
  root.traverse((child) => {
    if (child.isMesh) {
      meshCount += 1;
      child.material = solidMaterial(child.material);
      child.castShadow = true;
      child.receiveShadow = true;
      return;
    }
    if (child.isLine || child.isLineSegments || child.isLineLoop) {
      lineObjects.push(child);
    }
  });
  // Dense line-only overlays obscure the solid GLB meshes. Retain lines only
  // for assets that have no mesh geometry to render.
  if (meshCount > 0) {
    for (const line of lineObjects) line.visible = false;
  }
  return root;
}


function solidMaterial(material) {
  if (Array.isArray(material)) return material.map(solidMaterial);
  if (!material) {
    return new THREE.MeshStandardMaterial({ color: 0x96a6b4, roughness: 0.66, metalness: 0.08 });
  }
  if (material.wireframe) {
    material.wireframe = false;
  }
  // Untextured CAD exports often contain a white default material. Give those
  // surfaces a neutral metal finish while preserving authored colors/textures.
  const hsl = material.color?.getHSL?.({});
  if (!material.map && !material.vertexColors && hsl && hsl.s < 0.08 && hsl.l > 0.85) {
    return new THREE.MeshStandardMaterial({
      color: 0xaeb9c1,
      roughness: 0.4,
      metalness: 0.5,
      side: material.side,
      transparent: Boolean(material.transparent),
      opacity: Number.isFinite(material.opacity) ? material.opacity : 1,
    });
  }
  if (material.isMeshBasicMaterial && !material.map) {
    return new THREE.MeshStandardMaterial({
      color: material.color?.clone?.() || new THREE.Color(0x96a6b4),
      transparent: Boolean(material.transparent),
      opacity: Number.isFinite(material.opacity) ? material.opacity : 1,
      side: material.side,
      vertexColors: Boolean(material.vertexColors),
      roughness: 0.62,
      metalness: 0.08,
    });
  }
  material.needsUpdate = true;
  return material;
}


function fixedEquipmentPlacements(site) {
  return (site?.fixed_equipment_instances || [])
    .filter((item) => item?.instance_id)
    .map((item) => {
      const pose = item.pose || {};
      return {
        instance_id: item.instance_id,
        model_asset_id: item.model_asset_id || item.source_model_asset_id || null,
        dimensions_m: item.geometry?.dimensions_m || item.dimensions_m || {},
        position: {
          x_m: Number(pose.x_m || 0),
          y_m: Number(pose.y_m || 0),
          z_m: Number(pose.z_m || 0),
        },
        rotation: {
          roll_deg: Number(pose.roll_deg || 0),
          pitch_deg: Number(pose.pitch_deg || 0),
          yaw_deg: Number(pose.yaw_deg || 0),
        },
        fixed: true,
      };
    });
}


function fixedGeometryPrototype() {
  return new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial({
      color: 0x65727d,
      roughness: 0.78,
      metalness: 0.12,
    }),
  );
}


function instanceRenderRecord(placement, wrapper, renderSource, camera, canvas) {
  wrapper.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  let meshCount = 0;
  wrapper.traverse((child) => {
    if (child.isMesh && child.visible) meshCount += 1;
  });
  const bounds = new THREE.Box3().setFromObject(wrapper);
  const values = [
    bounds.min.x, bounds.min.y, bounds.min.z,
    bounds.max.x, bounds.max.y, bounds.max.z,
  ];
  const finiteWorldBounds = !bounds.isEmpty() && values.every(Number.isFinite);
  const visibility = finiteWorldBounds
    ? projectedBoundsRecord(bounds, camera, canvas)
    : {
      frustum_intersection: false,
      screen_projection_area_px2: 0,
      clipping_ratio: 1,
    };
  return {
    instance_id: placement.instance_id,
    model_asset_id: placement.model_asset_id || null,
    fixed: Boolean(placement.fixed),
    render_source: renderSource,
    mesh_count: meshCount,
    finite_world_bounds: finiteWorldBounds,
    world_bounds: {
      min: pointRecord(bounds.min, finiteWorldBounds),
      max: pointRecord(bounds.max, finiteWorldBounds),
    },
    ...visibility,
  };
}


function projectedBoundsRecord(bounds, camera, canvas) {
  const projectionView = new THREE.Matrix4().multiplyMatrices(
    camera.projectionMatrix,
    camera.matrixWorldInverse,
  );
  const frustum = new THREE.Frustum().setFromProjectionMatrix(projectionView);
  const intersects = frustum.intersectsBox(bounds);
  const corners = [];
  for (const x of [bounds.min.x, bounds.max.x]) {
    for (const y of [bounds.min.y, bounds.max.y]) {
      for (const z of [bounds.min.z, bounds.max.z]) {
        corners.push(new THREE.Vector3(x, y, z).project(camera));
      }
    }
  }
  const finite = corners.filter(
    (point) => Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z),
  );
  if (!intersects || !finite.length) {
    return {
      frustum_intersection: false,
      screen_projection_area_px2: 0,
      clipping_ratio: 1,
    };
  }
  const minX = Math.min(...finite.map((point) => point.x));
  const maxX = Math.max(...finite.map((point) => point.x));
  const minY = Math.min(...finite.map((point) => point.y));
  const maxY = Math.max(...finite.map((point) => point.y));
  const rawWidth = Math.max(0, maxX - minX);
  const rawHeight = Math.max(0, maxY - minY);
  const rawArea = rawWidth * rawHeight;
  const clippedWidth = Math.max(0, Math.min(1, maxX) - Math.max(-1, minX));
  const clippedHeight = Math.max(0, Math.min(1, maxY) - Math.max(-1, minY));
  const clippedArea = clippedWidth * clippedHeight;
  const canvasAreaScale = Math.max(1, Number(canvas.width || 0) * Number(canvas.height || 0) / 4);
  return {
    frustum_intersection: true,
    screen_projection_area_px2: clippedArea * canvasAreaScale,
    clipping_ratio: rawArea > 0
      ? Math.max(0, Math.min(1, 1 - clippedArea / rawArea))
      : 1,
  };
}


function pointRecord(point, finite) {
  return {
    x: finite ? Number(point.x) : null,
    y: finite ? Number(point.y) : null,
    z: finite ? Number(point.z) : null,
  };
}


function isReviewedFlowRelation(relation) {
  if (!relation || !relation.from_instance_id || !relation.to_instance_id) return false;
  if (relation.reviewed === true || relation.review_status === "approved") return true;
  const evidenceSource = String(
    relation.evidence_source
      || relation.source_type
      || relation.provenance?.source_type
      || "",
  ).toLowerCase();
  return new Set([
    "user_input",
    "user_explicit",
    "engineering_rag",
    "rag",
    "manual_review",
    "manual_override",
    "human_review",
    "expert_review",
  ]).has(evidenceSource);
}


function browserMetadata() {
  return {
    user_agent: navigator.userAgent || null,
    platform: navigator.platform || null,
    language: navigator.language || null,
    device_pixel_ratio: Number(window.devicePixelRatio || 1),
  };
}


function webglMetadata(gl) {
  const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
  return {
    version: safeGlParameter(gl, gl.VERSION),
    shading_language_version: safeGlParameter(gl, gl.SHADING_LANGUAGE_VERSION),
    vendor: debugInfo
      ? safeGlParameter(gl, debugInfo.UNMASKED_VENDOR_WEBGL)
      : safeGlParameter(gl, gl.VENDOR),
    renderer: debugInfo
      ? safeGlParameter(gl, debugInfo.UNMASKED_RENDERER_WEBGL)
      : safeGlParameter(gl, gl.RENDERER),
    context_lost: gl.isContextLost(),
    gl_error: Number(gl.getError()),
  };
}


function safeGlParameter(gl, parameter) {
  try {
    return String(gl.getParameter(parameter) || "") || null;
  } catch (_error) {
    return null;
  }
}


async function sha256Hex(bytes) {
  if (!globalThis.crypto?.subtle) return null;
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}


async function canvasPngSha256(canvas) {
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) return null;
  return sha256Hex(await blob.arrayBuffer());
}


function expectedViewportInstanceIds(layout) {
  if (!layout) return [];
  const ids = [
    ...(layout.placements || []).map((item) => item?.instance_id),
    ...(layout.site?.fixed_equipment_instances || []).map((item) => item?.instance_id),
  ].filter(Boolean).map(String);
  return [...new Set(ids)].sort();
}


function failedAssetReports(reports) {
  return (reports || [])
    .filter((item) => item?.status === "FAILED")
    .map((item) => ({
      model_asset_id: item.model_asset_id || null,
      attempts: Number(item.attempts || 0),
      error: item.error || "asset_load_failed",
    }))
    .sort(compareModelReports);
}


function emptyViewportVisibility(expectedInstanceIds) {
  return {
    intersecting_instance_ids: [],
    non_intersecting_instance_ids: [...expectedInstanceIds],
    total_screen_projection_area_px2: 0,
    maximum_clipping_ratio: 1,
  };
}


function viewportVisibility(records, expectedInstanceIds) {
  const intersecting = records
    .filter(
      (item) => item.frustum_intersection
        && Number(item.screen_projection_area_px2 || 0) > 0
        && Number(item.clipping_ratio ?? 1) < 1,
    )
    .map((item) => item.instance_id)
    .filter(Boolean);
  return {
    intersecting_instance_ids: [...new Set(intersecting)].sort(),
    non_intersecting_instance_ids: expectedInstanceIds.filter(
      (instanceId) => !intersecting.includes(instanceId),
    ),
    total_screen_projection_area_px2: records.reduce(
      (total, item) => total + Number(item.screen_projection_area_px2 || 0),
      0,
    ),
    maximum_clipping_ratio: records.length
      ? Math.max(...records.map((item) => Number(item.clipping_ratio ?? 1)))
      : 1,
  };
}


function nextFrame(draw) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      draw();
      resolve();
    };
    window.requestAnimationFrame(finish);
    window.setTimeout(finish, 100);
  });
}


function cloneJson(value) {
  if (typeof structuredClone === "function") return structuredClone(value);
  return JSON.parse(JSON.stringify(value));
}


function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}


function compareInstanceRecords(left, right) {
  return String(left.instance_id).localeCompare(String(right.instance_id));
}


function compareModelReports(left, right) {
  return String(left.model_asset_id).localeCompare(String(right.model_asset_id));
}


function createSiteGrid(site) {
  const points = [];
  const span = Math.max(site.width, site.depth);
  const step = Math.max(0.5, Math.ceil(span / 20));
  for (let x = -site.width / 2; x <= site.width / 2 + 1e-6; x += step) {
    points.push(
      new THREE.Vector3(x, 0, -site.depth / 2),
      new THREE.Vector3(x, 0, site.depth / 2),
    );
  }
  for (let z = -site.depth / 2; z <= site.depth / 2 + 1e-6; z += step) {
    points.push(
      new THREE.Vector3(-site.width / 2, 0, z),
      new THREE.Vector3(site.width / 2, 0, z),
    );
  }
  const material = new THREE.LineBasicMaterial({
    color: 0x526471,
    transparent: true,
    opacity: 0.32,
  });
  return new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points), material);
}

function floorText(text, width) {
  const canvas = document.createElement("canvas");
  canvas.width = 1024; canvas.height = 128;
  const ctx = canvas.getContext("2d");
  ctx.font = "600 62px Arial"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillStyle = "#e9ece6"; ctx.fillText(text, 512, 64, 995);
  const material = new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true, depthWrite: false });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, width / 8), material);
  mesh.rotation.x = -Math.PI / 2; return mesh;
}


function siteAnnotationRecords(siteData) {
  const regions = [];
  const sourceRegions = siteData?.placement_regions || siteData?.zones || [];
  const showNorthQuadrants = hasNorthFunctionalQuadrants(sourceRegions);
  for (const region of sourceRegions) {
    if (isWorkstationBayPlacementRegion(region)) continue;
    if (showNorthQuadrants && isBroadNorthParentRegion(region)) continue;
    const bounds = geometryBounds(region);
    if (!bounds) continue;
    regions.push({
      kind: "region",
      id: region.zone_id || region.region_id || null,
      label: semanticRegionLabel(region),
      bounds,
    });
  }
  const organized = siteData?.organized_workstations;
  const bays = siteData?.workstation_bays
    || organized?.workstation_bays
    || organized?.bays
    || (Array.isArray(organized) ? organized : []);
  const normalizedBays = [];
  for (const bay of bays || []) {
    const bounds = geometryBounds(bay);
    if (!bounds) continue;
    normalizedBays.push({
      kind: "bay",
      id: bay.bay_id || bay.id || null,
      label: String(bay.label || bay.bay_name || "Reserved bay"),
      bounds,
      reserved: bay.reserved === true,
    });
  }
  return { regions, bays: normalizedBays };
}


function isWorkstationBayPlacementRegion(region) {
  const id = String(region?.zone_id || region?.region_id || "").toLowerCase();
  return id.startsWith("workstation_bay_row_") || (id.includes("workstation_bay") && id.endsWith("/placement"));
}


function isBroadNorthParentRegion(region) {
  const identity = [region?.zone_id, region?.region_id, region?.zone_name, region?.label].filter(Boolean).join(" ").toLowerCase();
  const compact = identity.replace(/[^a-z0-9]/g, "");
  return compact.includes("northofeastwestaisle") || compact.includes("northoftheeastwestaisle");
}


function hasNorthFunctionalQuadrants(regions) {
  const identities = (regions || []).map((region) =>
    [region?.zone_id, region?.region_id, region?.zone_name, region?.label].filter(Boolean).join(" ").toLowerCase());
  const hasNorthwest = identities.some((value) => value.includes("northwest") || (value.includes("assembly") && value.includes("transfer")));
  const hasNortheast = identities.some((value) => value.includes("northeast") || (value.includes("test") && value.includes("finish")));
  return hasNorthwest && hasNortheast;
}


function geometryBounds(item) {
  const bounds = item?.bounds_m || item?.bounds || item?.operator_area || item || {};
  const minX = Number(bounds.min_x_m ?? bounds.min_x);
  const maxX = Number(bounds.max_x_m ?? bounds.max_x);
  const minZ = Number(bounds.min_z_m ?? bounds.min_z);
  const maxZ = Number(bounds.max_z_m ?? bounds.max_z);
  if ([minX, maxX, minZ, maxZ].every(Number.isFinite) && maxX > minX && maxZ > minZ) {
    return { minX, maxX, minZ, maxZ };
  }
  const pose = item?.pose;
  const dimensions = item?.dimensions_m;
  const x = Number(pose?.x_m), z = Number(pose?.z_m);
  const width = Number(dimensions?.width_m), depth = Number(dimensions?.depth_m);
  if (![x, z, width, depth].every(Number.isFinite) || width <= 0 || depth <= 0) return null;
  return { minX: x - width / 2, maxX: x + width / 2, minZ: z - depth / 2, maxZ: z + depth / 2 };
}


function semanticRegionLabel(region) {
  if (isWorkstationBayPlacementRegion(region)) return "";
  if (isBroadNorthParentRegion(region)) return "";
  const id = String(region?.zone_id || region?.region_id || "");
  const explicit = String(region?.label || region?.zone_name || region?.region_name || region?.name || "").trim();
  const direction = /northwest|north-west|\bnw\b/i.test(id) ? "NW"
    : /northeast|north-east|\bne\b/i.test(id) ? "NE"
      : /south/i.test(id) ? "South" : "";
  const role = /assembl|transfer/i.test(`${explicit} ${id}`) ? "Assembly & Transfer"
    : /test|finish/i.test(`${explicit} ${id}`) ? "Test & Finishing"
      : /machin|placement_south/i.test(`${explicit} ${id}`) ? "Machining" : "";
  if (explicit) {
    const title = explicit.replace(/[_/]+/g, " ").replace(/\s+/g, " ");
    const alreadyDirectional = direction === "NW" ? /north[- ]?west|\bnw\b/i.test(title)
      : direction === "NE" ? /north[- ]?east|\bne\b/i.test(title)
        : direction ? /\bsouth\b/i.test(title) : true;
    if (direction && !alreadyDirectional) return `${direction} · ${title}`;
    return title;
  }
  if (direction && role) return `${direction} · ${role}`;
  if (role) return role;
  return id.replace(/^(functional|placement)[/ _-]*/i, "").replace(/[_/]+/g, " ").trim();
}


function addFloorRegion(group, record, style) {
  const { minX, maxX, minZ, maxZ } = record.bounds;
  const width = maxX - minX, depth = maxZ - minZ;
  if (!(width > 0 && depth > 0)) return;
  if (!style.edgeOnly) {
    const patch = new THREE.Mesh(
      new THREE.PlaneGeometry(width, depth),
      new THREE.MeshStandardMaterial({
        color: style.fill,
        roughness: 0.96,
        transparent: true,
        opacity: style.fillOpacity,
        depthWrite: false,
        side: THREE.DoubleSide,
      }),
    );
    patch.rotation.x = -Math.PI / 2;
    patch.position.set((minX + maxX) / 2, 0.006, (minZ + maxZ) / 2);
    patch.renderOrder = 1;
    patch.userData.overlayKind = `${record.kind}-floor-patch`;
    group.add(patch);
  }
  const points = [
    new THREE.Vector3(minX, 0.035, minZ), new THREE.Vector3(maxX, 0.035, minZ),
    new THREE.Vector3(maxX, 0.035, maxZ), new THREE.Vector3(minX, 0.035, maxZ),
  ];
  const edge = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color: style.edge, transparent: true, opacity: style.edgeOpacity, depthWrite: false }),
  );
  edge.renderOrder = 2;
  edge.userData.overlayKind = `${record.kind}-boundary`;
  group.add(edge);
  if (record.label && (!record.kind.includes("bay") || record.reserved)) {
    const labelWidth = Math.max(0.8, Math.min(width * 0.84, record.reserved ? 4.5 : 9));
    const label = floorText(String(record.label).toUpperCase(), labelWidth);
    const labelZ = record.kind === "region" && /^(NW|NE|SOUTH)/i.test(record.label)
      ? maxZ - Math.min(0.7, depth * 0.12)
      : (minZ + maxZ) / 2;
    label.position.set((minX + maxX) / 2, 0.055, labelZ);
    label.material.color.set(style.labelColor || "#e0e7e2");
    label.renderOrder = 3;
    label.userData.overlayKind = `${record.kind}-label`;
    group.add(label);
  }
}


function addFloorHatch(group, bounds, color) {
  const { minX, maxX, minZ, maxZ } = bounds;
  const width = maxX - minX, depth = maxZ - minZ;
  const points = [];
  for (let offset = -depth; offset <= width; offset += 0.38) {
    const lo = Math.max(0, -offset), hi = Math.min(depth, width - offset);
    if (hi <= lo) continue;
    points.push(
      new THREE.Vector3(minX + lo + offset, 0.044, minZ + lo),
      new THREE.Vector3(minX + hi + offset, 0.044, minZ + hi),
    );
  }
  if (!points.length) return;
  const hatch = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.72, depthWrite: false }),
  );
  hatch.userData.overlayKind = "reserved-bay-hatch";
  group.add(hatch);
}


function compareLayoutChanges(beforeLayout, afterLayout) {
  const before = new Map((beforeLayout?.placements || []).filter((item) => item?.instance_id).map((item) => [item.instance_id, item]));
  const after = new Map((afterLayout?.placements || []).filter((item) => item?.instance_id).map((item) => [item.instance_id, item]));
  const ids = [...new Set([...before.keys(), ...after.keys()])].sort();
  return ids.map((instanceId) => {
    const previous = before.get(instanceId) || null;
    const current = after.get(instanceId) || null;
    const previousPosition = placementXZ(previous);
    const currentPosition = placementXZ(current);
    const dx = previous && current ? currentPosition.x - previousPosition.x : 0;
    const dz = previous && current ? currentPosition.z - previousPosition.z : 0;
    const moved = Boolean(previous && current && Math.hypot(dx, dz) > 1e-4);
    const previousYaw = placementYaw(previous), currentYaw = placementYaw(current);
    const yawDelta = signedAngleDelta(previousYaw, currentYaw);
    const rotated = Boolean(previous && current && Math.abs(yawDelta) > 1e-3);
    const previousFootprint = placementBounds(previous);
    const currentFootprint = placementBounds(current);
    const resized = Boolean(previous && current && boundsDiffer(previousFootprint, currentFootprint));
    const modelChanged = Boolean(previous && current && modelIdentity(previous) !== modelIdentity(current));
    const changed = !previous || !current || moved || rotated || resized || modelChanged;
    return {
      instanceId,
      previous,
      current,
      previousPosition,
      currentPosition,
      previousFootprint,
      currentFootprint,
      dx,
      dz,
      yawDelta,
      moved,
      rotated,
      resized,
      modelChanged,
      added: !previous || modelChanged,
      removed: !current || modelChanged,
      changed,
    };
  });
}


function renderLayoutChangeOverlays(group, changes) {
  const changed = changes.filter((item) => item.changed);
  const summary = {
    changed_count: changed.length,
    moved_count: changed.filter((item) => item.moved).length,
    rotated_count: changed.filter((item) => item.rotated).length,
    resized_count: changed.filter((item) => item.resized).length,
    added_count: changed.filter((item) => item.added).length,
    removed_count: changed.filter((item) => item.removed).length,
  };
  for (const item of changed) {
    if (item.previous && item.previousFootprint) {
      addFootprintOutline(group, item.previousFootprint, 0x5cabb0, true, `before-footprint:${item.instanceId}`);
    }
    if (item.current && item.currentFootprint) {
      addFootprintOutline(group, item.currentFootprint, item.added ? 0x7dc4b3 : 0xe1b270, false, `after-footprint:${item.instanceId}`);
    }
    if (item.added && item.currentPosition) addChangeMarker(group, item.currentPosition, "+", 0x72c5a8, `added:${item.instanceId}`);
    if (item.removed && item.previousPosition) addChangeMarker(group, item.previousPosition, "−", 0xd98582, `removed:${item.instanceId}`);
    if (item.moved) addDisplacementArrow(group, item.previousPosition, item.currentPosition, item.instanceId);
    if (item.rotated && !item.moved) addRotationArrow(group, item.currentPosition, item.current, item.yawDelta, item.instanceId);
    else if (item.rotated && item.moved) addRotationArrow(group, item.currentPosition, item.current, item.yawDelta, item.instanceId);
  }
  group.userData.changeSummary = summary;
  return summary;
}


function addFootprintOutline(group, bounds, color, dashed, name) {
  const points = [
    new THREE.Vector3(bounds.minX, 0.16, bounds.minZ),
    new THREE.Vector3(bounds.maxX, 0.16, bounds.minZ),
    new THREE.Vector3(bounds.maxX, 0.16, bounds.maxZ),
    new THREE.Vector3(bounds.minX, 0.16, bounds.maxZ),
    new THREE.Vector3(bounds.minX, 0.16, bounds.minZ),
  ];
  const material = dashed
    ? new THREE.LineDashedMaterial({ color, dashSize: 0.26, gapSize: 0.16, transparent: true, opacity: 0.92 })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.98 });
  const outline = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), material);
  if (dashed) outline.computeLineDistances();
  outline.userData.overlayKind = dashed ? "before-footprint" : "after-footprint";
  outline.userData.instanceId = name.split(":").at(-1);
  group.add(outline);
}


function addDisplacementArrow(group, from, to, instanceId) {
  const direction = new THREE.Vector3(to.x - from.x, 0, to.z - from.z);
  const length = direction.length();
  if (!(length > 1e-4)) return;
  direction.normalize();
  const arrow = new THREE.ArrowHelper(
    direction,
    new THREE.Vector3(from.x, 0.24, from.z),
    length,
    0x79bdd0,
    Math.min(0.72, Math.max(0.32, length * 0.18)),
    Math.min(0.38, Math.max(0.18, length * 0.09)),
  );
  arrow.userData.overlayKind = "displacement-arrow";
  arrow.userData.instanceId = instanceId;
  group.add(arrow);
}


function addRotationArrow(group, center, current, yawDelta, instanceId) {
  const bounds = placementBounds(current);
  const radius = Math.max(0.38, Math.min(0.9, Math.min(bounds.maxX - bounds.minX, bounds.maxZ - bounds.minZ) * 0.42));
  const startYaw = placementYaw(current) - yawDelta;
  const steps = Math.max(8, Math.ceil(Math.abs(yawDelta) / 9));
  const points = [];
  for (let index = 0; index <= steps; index += 1) {
    const yaw = startYaw + yawDelta * (index / steps);
    const radians = THREE.MathUtils.degToRad(yaw);
    points.push(new THREE.Vector3(center.x - Math.sin(radians) * radius, 0.3, center.z + Math.cos(radians) * radius));
  }
  const arc = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color: 0xdaa65d, transparent: true, opacity: 0.96 }),
  );
  arc.userData.overlayKind = "rotation-arc";
  arc.userData.instanceId = instanceId;
  group.add(arc);
  const tangent = points.at(-1).clone().sub(points.at(-2)).normalize();
  const head = new THREE.ArrowHelper(tangent, points.at(-1), 0.38, 0xdaa65d, 0.27, 0.16);
  head.userData.overlayKind = "rotation-arrowhead";
  head.userData.instanceId = instanceId;
  group.add(head);
}


function addChangeMarker(group, center, symbol, color, name) {
  const size = 0.27;
  const points = symbol === "+"
    ? [new THREE.Vector3(center.x - size, 0.34, center.z), new THREE.Vector3(center.x + size, 0.34, center.z),
      new THREE.Vector3(center.x, 0.34, center.z - size), new THREE.Vector3(center.x, 0.34, center.z + size)]
    : [new THREE.Vector3(center.x - size, 0.34, center.z), new THREE.Vector3(center.x + size, 0.34, center.z)];
  const marker = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color, linewidth: 2 }),
  );
  marker.userData.overlayKind = symbol === "+" ? "added-marker" : "removed-marker";
  marker.userData.instanceId = name.split(":").at(-1);
  group.add(marker);
}


function placementXZ(placement) {
  return { x: Number(placement?.position?.x_m || 0), z: Number(placement?.position?.z_m || 0) };
}


function placementYaw(placement) {
  return Number(placement?.rotation?.yaw_deg ?? placement?.yaw_deg ?? 0);
}


function signedAngleDelta(from, to) {
  let delta = ((to - from + 180) % 360 + 360) % 360 - 180;
  if (Math.abs(delta + 180) < 1e-8) delta = 180;
  return delta;
}


function placementBounds(placement) {
  const explicit = geometryBounds(placement?.footprint || {});
  if (explicit) return explicit;
  const center = placementXZ(placement);
  const width = Math.max(0.05, Number(placement?.dimensions_m?.width_m || 0.05));
  const depth = Math.max(0.05, Number(placement?.dimensions_m?.depth_m || 0.05));
  return { minX: center.x - width / 2, maxX: center.x + width / 2, minZ: center.z - depth / 2, maxZ: center.z + depth / 2 };
}


function boundsDiffer(left, right, tolerance = 1e-4) {
  if (!left || !right) return true;
  const leftSize = [left.maxX - left.minX, left.maxZ - left.minZ];
  const rightSize = [right.maxX - right.minX, right.maxZ - right.minZ];
  return leftSize.some((value, index) => Math.abs(value - rightSize[index]) > tolerance);
}


function modelIdentity(placement) {
  if (!placement) return "";
  if (placement.model_asset_id) return String(placement.model_asset_id);
  if (Array.isArray(placement.compound_members)) {
    return placement.compound_members.map((member) => member?.model_asset_id || member?.instance_id || "").sort().join("|");
  }
  return String(placement.equipment_type || placement.name || "");
}


function disposeObject(root) {
  if (!root) return;
  root.traverse((child) => {
    child.geometry?.dispose?.();
    if (Array.isArray(child.material)) {
      child.material.forEach((material) => material.dispose?.());
    } else {
      child.material?.dispose?.();
    }
  });
}


function siteDimensions(site) {
  const dimensions = site?.dimensions_m || site || {};
  return {
    width: Math.max(1, Number(dimensions.width_m || site?.width_m || 1)),
    depth: Math.max(1, Number(dimensions.depth_m || site?.depth_m || 1)),
  };
}


function sceneBoundsForCamera(root, site) {
  root?.updateMatrixWorld(true);
  return normalizedCameraBounds(
    root ? new THREE.Box3().setFromObject(root) : null,
    site,
  );
}


function normalizedCameraBounds(bounds, site) {
  const values = bounds
    ? [
      bounds.min.x, bounds.min.y, bounds.min.z,
      bounds.max.x, bounds.max.y, bounds.max.z,
    ]
    : [];
  if (bounds && !bounds.isEmpty() && values.every(Number.isFinite)) return bounds.clone();
  const width = Math.max(1, Number(site?.width || 1));
  const depth = Math.max(1, Number(site?.depth || 1));
  return new THREE.Box3().setFromCenterAndSize(
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(width, 1, depth),
  );
}


function placementCenter(placement, y) {
  return new THREE.Vector3(
    Number(placement.position?.x_m || 0),
    y,
    Number(placement.position?.z_m || 0),
  );
}


function overlapsAny(placement, placements) {
  const left = footprint(placement);
  if (!left) return false;
  return placements.some((other) => {
    if (other === placement) return false;
    const right = footprint(other);
    return right && left.minX < right.maxX && left.maxX > right.minX
      && left.minZ < right.maxZ && left.maxZ > right.minZ;
  });
}


function footprint(placement) {
  if (placement.footprint) {
    return {
      minX: Number(placement.footprint.min_x),
      maxX: Number(placement.footprint.max_x),
      minZ: Number(placement.footprint.min_z),
      maxZ: Number(placement.footprint.max_z),
    };
  }
  const width = Number(placement.dimensions_m?.width_m || 0);
  const depth = Number(placement.dimensions_m?.depth_m || 0);
  if (!(width > 0 && depth > 0)) return null;
  const x = Number(placement.position?.x_m || 0);
  const z = Number(placement.position?.z_m || 0);
  return { minX: x - width / 2, maxX: x + width / 2, minZ: z - depth / 2, maxZ: z + depth / 2 };
}
