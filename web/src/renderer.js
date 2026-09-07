import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const SIZE = 180,
  ALTITUDE = 37;
const palettes = [
  {
    sky: "#557c98",
    horizon: "#e5dcc5",
    fog: "#b5c9bb",
    water: "#2c6e69",
    grass: "#60724b",
    rock: "#858378",
    snow: "#ebece1",
    tree: "#254839",
    sun: "#fff1cf",
  },
  {
    sky: "#799eaf",
    horizon: "#f3cf9b",
    fog: "#d9b789",
    water: "#628c74",
    grass: "#927e47",
    rock: "#b88756",
    snow: "#dab688",
    tree: "#726c37",
    sun: "#ffe0ad",
  },
  {
    sky: "#333451",
    horizon: "#af8a99",
    fog: "#716e91",
    water: "#518b9f",
    grass: "#767286",
    rock: "#746684",
    snow: "#a4b7b9",
    tree: "#48a299",
    sun: "#eedafb",
  },
];
function randomGenerator(seed) {
  let a = seed | 0;
  return () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
// Interpolation only samples the model's spatial fields. It adds no replacement terrain.
function sample(field, u, v) {
  const x = THREE.MathUtils.clamp(u, 0, 1) * 63,
    z = THREE.MathUtils.clamp(v, 0, 1) * 63,
    x0 = Math.floor(x),
    z0 = Math.floor(z),
    x1 = Math.min(63, x0 + 1),
    z1 = Math.min(63, z0 + 1);
  return THREE.MathUtils.lerp(
    THREE.MathUtils.lerp(field[z0][x0], field[z0][x1], x - x0),
    THREE.MathUtils.lerp(field[z1][x0], field[z1][x1], x - x0),
    z - z0,
  );
}
function heightGeometry(state, segments = 190, water = false) {
  const p = [],
    uv = [],
    colors = [],
    idx = [],
    depths = [];
  const palette = palettes[state.biome],
    grass = new THREE.Color(palette.grass),
    rock = new THREE.Color(palette.rock),
    snow = new THREE.Color(palette.snow),
    sand = new THREE.Color(state.biome === 1 ? "#d2ad74" : "#768269"),
    burn = new THREE.Color("#6a4436");
  for (let z = 0; z <= segments; z++)
    for (let x = 0; x <= segments; x++) {
      const u = x / segments,
        v = z / segments,
        h = sample(state.height, u, v),
        w = sample(state.ecology[0], u, v),
        veg = sample(state.ecology[1], u, v),
        heat = sample(state.ecology[2], u, v),
        ground = h * ALTITUDE;
      // Water is a visualization of the saved water field, not a fluid simulation.
      // Match the terrain triangles and keep every water vertex above the ground.
      // The smooth maximum removes the old abrupt switch at the basin boundary.
      const baseline = Math.max(0, (-h - 0.1) * 0.6),
        level = -ALTITUDE * 0.1 + (w - baseline) * 8,
        film = ground + 0.045 + w * 2;
      const blend = Math.max(0.18 - Math.abs(level - film), 0) / 0.18;
      const surface = Math.max(level, film) + blend * blend * 0.045;
      p.push((u - 0.5) * SIZE, water ? surface : ground, (v - 0.5) * SIZE);
      uv.push(u, v);
      depths.push(surface - ground);
      const dx =
          sample(state.height, u + 0.008, v) -
          sample(state.height, u - 0.008, v),
        dz =
          sample(state.height, u, v + 0.008) -
          sample(state.height, u, v - 0.008),
        slope = Math.min(1, Math.hypot(dx, dz) * 8.5);
      const c = sand
        .clone()
        .lerp(grass, THREE.MathUtils.smoothstep(veg, 0.06, 0.65));
      c.lerp(rock, THREE.MathUtils.smoothstep(slope, 0.36, 0.88));
      if (state.biome === 0)
        c.lerp(
          snow,
          THREE.MathUtils.smoothstep(h, 0.57, 0.81) * (1 - slope * 0.55),
        );
      if (state.biome === 1) c.lerp(sand, Math.max(0, h + 0.15) * 0.45);
      c.lerp(burn, heat * 0.46);
      c.multiplyScalar(1 - w * 0.24);
      colors.push(c.r, c.g, c.b);
    }
  for (let z = 0; z < segments; z++)
    for (let x = 0; x < segments; x++) {
      const a = z * (segments + 1) + x,
        b = a + 1,
        c = a + segments + 1,
        d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  g.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  if (water)
    g.setAttribute("waterDepth", new THREE.Float32BufferAttribute(depths, 1));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}
const noiseGLSL = `float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);} float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);} float fbm(vec2 p){return noise(p)*.5+noise(p*2.03)*.25+noise(p*4.01)*.125+noise(p*8.03)*.0625;}`;

export class WorldRenderer {
  constructor(canvas, { onPick, onFps, onDirection, onMode } = {}) {
    this.canvas = canvas;
    this.onPick = onPick;
    this.onFps = onFps;
    this.onDirection = onDirection;
    this.onMode = onMode;
    this.mode = "orbit";
    this.brush = "";
    this.radius = 0.07;
    this.keys = new Set();
    this.state = null;
    this.last = performance.now();
    this.frames = 0;
    this.fpsStart = this.last;
    this.time = 0;
    this.drag = null;
    this.mouse = new THREE.Vector2();
    this.ray = new THREE.Raycaster();
    this.group = new THREE.Group();
    this.resources = [];
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.6));
    this.renderer.setSize(innerWidth, innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.02;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(palettes[0].fog, 105, 260);
    this.scene.add(this.group);
    this.camera = new THREE.PerspectiveCamera(
      49,
      innerWidth / innerHeight,
      0.15,
      3000,
    );
    this.camera.position.set(52, 30, 68);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.065;
    this.controls.minDistance = 8;
    this.controls.maxDistance = 175;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.minPolarAngle = 0.1;
    this.controls.enablePan = true;
    this.controls.screenSpacePanning = false;
    this.controls.rotateSpeed = 0.5;
    this.controls.zoomSpeed = 0.8;
    this.sun = new THREE.DirectionalLight("#fff1d2", 2.4);
    this.sun.position.set(-88, 110, -65);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    Object.assign(this.sun.shadow.camera, {
      left: -125,
      right: 125,
      top: 125,
      bottom: -125,
      near: 1,
      far: 350,
    });
    this.sun.shadow.bias = -0.0003;
    this.sun.shadow.normalBias = 0.5;
    this.sun.shadow.radius = 3;
    this.scene.add(this.sun);
    this.fill = new THREE.HemisphereLight("#d1e3ea", "#777363", 1.25);
    this.scene.add(this.fill);
    this.makeSky();
    this.makeRain();
    this.makeBrush();
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(
      new THREE.Vector2(innerWidth, innerHeight),
      0.19,
      0.55,
      1.3,
    );
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.bind();
    this.resize();
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }
  makeSky() {
    this.skyUniforms = {
      time: { value: 0 },
      top: { value: new THREE.Color(palettes[0].sky) },
      horizon: { value: new THREE.Color(palettes[0].horizon) },
      sunColor: { value: new THREE.Color(palettes[0].sun) },
      sunDirection: { value: this.sun.position.clone().normalize() },
      alien: { value: 0 },
      rain: { value: 0 },
    };
    const mat = new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      uniforms: this.skyUniforms,
      vertexShader: `varying vec3 vWorld; void main(){vWorld=(modelMatrix*vec4(position,1.)).xyz;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
      fragmentShader: `uniform float time;uniform float alien;uniform float rain;uniform vec3 top;uniform vec3 horizon;uniform vec3 sunColor;uniform vec3 sunDirection;varying vec3 vWorld;${noiseGLSL}void main(){vec3 d=normalize(vWorld-cameraPosition);float elev=max(d.y,0.);vec3 color=mix(horizon,top,pow(elev,.4));float glow=pow(max(dot(d,sunDirection),0.),12.);float disk=smoothstep(.9991,.9997,dot(d,sunDirection));color+=sunColor*(glow*.23+disk*3.);vec2 p=d.xz/(max(.06,d.y)+.18)*1.5;float clouds=smoothstep(.48,.69,fbm(p+vec2(time*.004,0.)));clouds*=smoothstep(.02,.2,d.y)*(1.-smoothstep(.68,1.,d.y));color=mix(color,mix(vec3(.95,.94,.86),vec3(.47,.53,.55),rain),clouds*.42);color=mix(color,vec3(.42,.48,.51),rain*.24);float stars=step(.997,hash(floor(d.xz/(d.y+.4)*850.)))*smoothstep(.25,.8,d.y);color+=stars*alien*.4;gl_FragColor=vec4(color,1.);\n#include <tonemapping_fragment>\n#include <colorspace_fragment>}`,
    });
    this.sky = new THREE.Mesh(new THREE.SphereGeometry(1600, 32, 16), mat);
    this.sky.renderOrder = -10;
    this.scene.add(this.sky);
  }
  makeRain() {
    const p = [];
    const rnd = randomGenerator(728);
    for (let i = 0; i < 1300; i++) {
      const x = (rnd() - 0.5) * SIZE,
        y = rnd() * 85 + 8,
        z = (rnd() - 0.5) * SIZE;
      p.push(x, y, z, x - 0.16, y - 1.2 - rnd() * 0.7, z + 0.05);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
    this.rain = new THREE.LineSegments(
      g,
      new THREE.LineBasicMaterial({
        color: "#c5d5d7",
        transparent: true,
        opacity: 0,
        depthWrite: false,
      }),
    );
    this.rain.visible = false;
    this.scene.add(this.rain);
  }
  makeBrush() {
    const pts = [];
    for (let i = 0; i <= 96; i++)
      pts.push(
        new THREE.Vector3(
          Math.cos((i / 96) * Math.PI * 2),
          0,
          Math.sin((i / 96) * Math.PI * 2),
        ),
      );
    this.brushRing = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({
        color: "#f4edc6",
        transparent: true,
        opacity: 0.9,
        depthTest: false,
      }),
    );
    this.brushRing.renderOrder = 20;
    this.brushRing.visible = false;
    this.scene.add(this.brushRing);
  }
  terrainMaterial() {
    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.96,
      metalness: 0.025,
    });
    mat.onBeforeCompile = (shader) => {
      shader.vertexShader =
        "varying vec3 detailPosition;\n" + shader.vertexShader;
      shader.vertexShader = shader.vertexShader.replace(
        "#include <begin_vertex>",
        "#include <begin_vertex>\ndetailPosition=position;",
      );
      shader.fragmentShader =
        "varying vec3 detailPosition;\n" +
        noiseGLSL +
        "\n" +
        shader.fragmentShader;
      shader.fragmentShader = shader.fragmentShader.replace(
        "#include <color_fragment>",
        `#include <color_fragment>\nfloat grain=fbm(detailPosition.xz*2.4)+noise(detailPosition.xz*14.)*.12;float strata=sin(detailPosition.y*2.+fbm(detailPosition.xz*.25)*5.)*.027;diffuseColor.rgb*=.83+grain*.36+strata;`,
      );
    };
    return mat;
  }
  waterMaterial(state) {
    const data = new Float32Array(64 * 64);
    for (let z = 0; z < 64; z++)
      for (let x = 0; x < 64; x++) data[z * 64 + x] = state.ecology[0][z][x];
    const texture = new THREE.DataTexture(
      data,
      64,
      64,
      THREE.RedFormat,
      THREE.FloatType,
    );
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
    this.resources.push(texture);
    const p = palettes[state.biome];
    const uniforms = {
      time: { value: this.time },
      waterColor: { value: new THREE.Color(p.water) },
      skyColor: { value: new THREE.Color(p.horizon) },
      skyTop: { value: new THREE.Color(p.sky) },
      sunColor: { value: new THREE.Color(p.sun) },
      waterMap: { value: texture },
      sunDirection: { value: this.sun.position.clone().normalize() },
      fogColor: { value: new THREE.Color(p.fog) },
    };
    this.waterUniforms = uniforms;
    return new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      uniforms,
      vertexShader: `
        attribute float waterDepth;
        varying vec3 vWorld;
        varying vec2 vUv;
        varying float vDepth;
        void main(){
          vUv=uv;vDepth=waterDepth;
          vWorld=(modelMatrix*vec4(position,1.)).xyz;
          gl_Position=projectionMatrix*viewMatrix*vec4(vWorld,1.);
        }`,
      fragmentShader: `
        uniform float time;
        uniform vec3 waterColor,skyColor,skyTop,sunColor,sunDirection,fogColor;
        uniform sampler2D waterMap;
        varying vec3 vWorld;
        varying vec2 vUv;
        varying float vDepth;
        ${noiseGLSL}
        void main(){
          // Sample texel centers to match the CPU's interpolation of the 64x64 field.
          float w=texture2D(waterMap,(vUv*63.+.5)/64.).r;
          float shore=smoothstep(.001,.025,w);
          if(shore<.001)discard;
          vec2 p=vWorld.xz;
          // Animate the surface normal only. Moving shoreline vertices caused
          // depth intersections and exposed the triangular mesh underneath.
          vec2 gradient=vec2(.034,.018)*cos(dot(p,vec2(.37,.19))-time*.72);
          gradient+=vec2(-.021,.039)*cos(dot(p,vec2(-.28,.53))+time*.93);
          gradient+=vec2(.013,-.01)*cos(dot(p,vec2(1.11,-.87))-time*1.21);
          vec3 surfaceNormal=normalize(cross(dFdx(vWorld),dFdy(vWorld)));
          if(surfaceNormal.y<0.)surfaceNormal=-surfaceNormal;
          vec3 normal=normalize(surfaceNormal+vec3(-gradient.x,0.,-gradient.y));
          vec3 eye=normalize(cameraPosition-vWorld);
          float fresnel=.035+.965*pow(1.-max(dot(eye,normal),0.),5.);
          vec3 reflectedView=reflect(-eye,normal);
          vec3 reflection=mix(skyColor,skyTop,pow(max(reflectedView.y,0.),.4));
          float deep=1.-exp(-max(vDepth,0.)*.16);
          vec3 body=waterColor*mix(1.3,.69,deep);
          vec3 color=mix(body,reflection,fresnel*.86+.07);
          vec3 halfDirection=normalize(sunDirection+eye);
          float highlight=max(dot(normal,halfDirection),0.);
          color+=sunColor*(pow(highlight,170.)*.65+pow(highlight,36.)*.07);
          float foam=(1.-smoothstep(.08,.45,vDepth))*noise(p*.75+time*.04)*.08;
          color=mix(color,mix(waterColor,skyColor,.55),foam);
          float fog=smoothstep(105.,260.,length(cameraPosition-vWorld));
          color=mix(color,fogColor,fog);
          gl_FragColor=vec4(color,shore*mix(.37,.9,deep));
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    });
  }
  clearWorld() {
    this.group.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        for (const m of Array.isArray(o.material) ? o.material : [o.material])
          m.dispose();
      }
    });
    this.group.clear();
    for (const r of this.resources) r.dispose();
    this.resources = [];
  }
  setWorld(state, { resetCamera = false } = {}) {
    this.clearWorld();
    this.state = state;
    const palette = palettes[state.biome];
    this.skyUniforms.top.value.set(palette.sky);
    this.skyUniforms.horizon.value.set(palette.horizon);
    this.skyUniforms.sunColor.value.set(palette.sun);
    this.skyUniforms.alien.value = state.biome === 2 ? 1 : 0;
    this.skyUniforms.rain.value = state.weather.rain;
    this.scene.fog.color.set(palette.fog);
    this.sun.color.set(palette.sun);
    this.sun.intensity = state.biome === 2 ? 1.8 : 2.4;
    this.fill.color.set(state.biome === 2 ? "#b7bbda" : "#d1e3ea");
    this.rain.material.opacity = state.weather.rain * 0.33;
    this.rain.visible = state.weather.rain > 0.05;
    this.terrain = new THREE.Mesh(
      heightGeometry(state),
      this.terrainMaterial(),
    );
    this.terrain.castShadow = true;
    this.terrain.receiveShadow = true;
    this.group.add(this.terrain);
    const water = new THREE.Mesh(
      heightGeometry(state, 190, true),
      this.waterMaterial(state),
    );
    water.renderOrder = 2;
    this.group.add(water);
    this.makeSkirt();
    this.makeVegetation();
    this.makeRocks();
    this.makeGrass();
    if (resetCamera) this.home();
    this.updateBrush();
  }
  makeSkirt() {
    const state = this.state,
      points = [];
    for (let i = 0; i < 64; i++) points.push([i / 63, 0]);
    for (let i = 1; i < 64; i++) points.push([1, i / 63]);
    for (let i = 62; i >= 0; i--) points.push([i / 63, 1]);
    for (let i = 62; i > 0; i--) points.push([0, i / 63]);
    const p = [],
      idx = [];
    for (const [u, v] of points) {
      p.push(
        (u - 0.5) * SIZE,
        sample(state.height, u, v) * ALTITUDE - 0.015,
        (v - 0.5) * SIZE,
        (u - 0.5) * SIZE,
        -46,
        (v - 0.5) * SIZE,
      );
    }
    for (let i = 0; i < points.length; i++) {
      let a = i * 2,
        b = ((i + 1) % points.length) * 2;
      idx.push(a, a + 1, b, b, a + 1, b + 1);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    const m = new THREE.MeshStandardMaterial({
      color: palettes[state.biome].rock,
      roughness: 1,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(g, m);
    mesh.receiveShadow = true;
    this.group.add(mesh);
    const base = new THREE.Mesh(
      new THREE.PlaneGeometry(1400, 1400),
      new THREE.MeshBasicMaterial({ color: palettes[state.biome].fog }),
    );
    base.rotation.x = -Math.PI / 2;
    base.position.y = -46;
    this.group.add(base);
  }
  candidatePoints(count, salt) {
    const r = randomGenerator(this.state.seed ^ salt),
      out = [];
    for (let i = 0; i < count; i++) {
      const u = 0.025 + r() * 0.95,
        v = 0.025 + r() * 0.95,
        h = sample(this.state.height, u, v),
        veg = sample(this.state.ecology[1], u, v),
        water = sample(this.state.ecology[0], u, v),
        heat = sample(this.state.ecology[2], u, v),
        slope =
          Math.hypot(
            sample(this.state.height, u + 0.006, v) -
              sample(this.state.height, u - 0.006, v),
            sample(this.state.height, u, v + 0.006) -
              sample(this.state.height, u, v - 0.006),
          ) * 11;
      out.push({
        u,
        v,
        h,
        veg,
        water,
        heat,
        slope,
        random: r(),
        scale: r(),
        rot: r() * Math.PI * 2,
      });
    }
    return out;
  }
  addInstances(geometry, material, points, transform, colorFunction) {
    const mesh = new THREE.InstancedMesh(geometry, material, points.length);
    const o = new THREE.Object3D();
    points.forEach((p, i) => {
      o.position.set((p.u - 0.5) * SIZE, p.h * ALTITUDE, (p.v - 0.5) * SIZE);
      o.rotation.set(0, p.rot, 0);
      o.scale.set(1, 1, 1);
      transform(o, p, i);
      o.updateMatrix();
      mesh.setMatrixAt(i, o.matrix);
      if (colorFunction) mesh.setColorAt(i, colorFunction(p, i));
    });
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
    this.group.add(mesh);
    return mesh;
  }
  makeVegetation() {
    const state = this.state,
      biome = state.biome;
    const pts = this.candidatePoints(6100, 5241).filter(
      (p) =>
        p.veg > 0.27 &&
        p.random < p.veg * (biome === 1 ? 0.37 : 0.78) &&
        p.water < 0.008 &&
        p.slope < 0.69 &&
        p.h < (biome === 0 ? 0.64 : 0.92),
    );
    if (biome === 0) {
      const trunk = new THREE.CylinderGeometry(0.045, 0.1, 1.6, 5);
      trunk.translate(0, 0.8, 0);
      const cones = [];
      for (let i = 0; i < 4; i++) {
        const g = new THREE.ConeGeometry(0.63 - i * 0.115, 1.25 - i * 0.19, 7);
        g.translate(0, 1.0 + i * 0.38, 0);
        cones.push(g);
      }
      const leaves = mergeGeometries(cones);
      cones.forEach((g) => g.dispose());
      const transform = (o, p) => {
        const s = 0.9 + p.scale * 1.7;
        o.scale.set(s, s * (0.9 + p.veg * 0.7), s);
      };
      this.addInstances(
        trunk,
        new THREE.MeshStandardMaterial({ color: "#675946", roughness: 1 }),
        pts,
        transform,
      );
      this.addInstances(
        leaves,
        new THREE.MeshStandardMaterial({ color: "#8ea489", roughness: 0.97 }),
        pts,
        transform,
        (p) =>
          new THREE.Color()
            .setHSL(
              0.32 + p.scale * 0.055,
              0.2 + p.veg * 0.19,
              0.15 + p.scale * 0.1,
            )
            .lerp(new THREE.Color("#76503b"), p.heat * 0.5),
      );
    } else if (biome === 1) {
      const g = new THREE.CylinderGeometry(0.15, 0.2, 2.2, 7, 3);
      g.translate(0, 1.1, 0);
      this.addInstances(
        g,
        new THREE.MeshStandardMaterial({ color: "#8d9866", roughness: 0.93 }),
        pts,
        (o, p) => {
          const s = 0.6 + p.scale * 1.2;
          o.scale.set(s, s, s);
        },
        (p) => new THREE.Color().setHSL(0.15, 0.19, 0.25 + p.scale * 0.18),
      );
      const cap = new THREE.SphereGeometry(0.32, 5, 4);
      cap.translate(0, 1.7, 0);
      this.addInstances(
        cap,
        new THREE.MeshStandardMaterial({ color: "#78804e", roughness: 1 }),
        pts,
        (o, p) => {
          const s = 0.6 + p.scale * 1.2;
          o.scale.set(s, s * 1.3, s);
        },
      );
    } else {
      const g = new THREE.OctahedronGeometry(0.6, 0);
      g.translate(0, 0.6, 0);
      this.addInstances(
        g,
        new THREE.MeshStandardMaterial({
          color: "#77d1ce",
          emissive: "#176570",
          emissiveIntensity: 0.32,
          roughness: 0.28,
          metalness: 0.25,
        }),
        pts,
        (o, p) => {
          o.scale.set(0.7 + p.scale * 0.8, 1.5 + p.scale * 4, 1);
          o.rotation.z = (p.scale - 0.5) * 0.26;
        },
        (p) => new THREE.Color().setHSL(0.47 + p.scale * 0.2, 0.45, 0.45),
      );
    }
  }
  makeRocks() {
    const biome = this.state.biome,
      pts = this.candidatePoints(2900, 6917).filter(
        (p) => p.water < 0.12 && p.random < 0.2 + p.slope * 0.7,
      );
    const geom = new THREE.DodecahedronGeometry(0.53, 0);
    const mat = new THREE.MeshStandardMaterial({
      color: palettes[biome].rock,
      roughness: 0.98,
      flatShading: true,
    });
    this.addInstances(
      geom,
      mat,
      pts,
      (o, p) => {
        const s = 0.25 + p.scale * 1.5;
        o.position.y += s * 0.14;
        o.scale.set(s * (1.1 + p.scale), s * (0.6 + p.scale * 0.55), s * 0.84);
        o.rotation.set(p.rot * 0.2, p.rot, p.scale * 0.7);
      },
      (p) =>
        new THREE.Color(
          p.h > 0.5 && biome === 0 ? "#d4d8cb" : palettes[biome].rock,
        ).multiplyScalar(0.63 + p.scale * 0.4),
    );
  }
  makeGrass() {
    const state = this.state,
      pts = this.candidatePoints(22000, 6110).filter(
        (p) =>
          p.water < 0.008 &&
          p.veg > 0.31 &&
          p.random < p.veg * 0.8 &&
          p.h < (state.biome === 0 ? 0.41 : 0.92) &&
          p.slope < 0.62,
      );
    const blade = new THREE.BufferGeometry();
    blade.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(
        [
          -0.15, 0, 0, 0.15, 0, 0, 0.06, 0.65, 0, 0, 0, -0.15, 0, 0, 0.15, 0,
          0.52, 0.07,
        ],
        3,
      ),
    );
    blade.setIndex([0, 1, 2, 3, 4, 5]);
    blade.computeVertexNormals();
    this.addInstances(
      blade,
      new THREE.MeshStandardMaterial({
        color: palettes[state.biome].grass,
        roughness: 1,
        side: THREE.DoubleSide,
      }),
      pts,
      (o, p) => {
        const s = 0.35 + p.scale * 0.9;
        o.scale.set(s, s * (0.5 + p.veg), s);
      },
      (p) =>
        new THREE.Color(palettes[state.biome].grass)
          .lerp(new THREE.Color("#b0a26a"), p.scale * 0.65)
          .lerp(new THREE.Color("#7b4833"), p.heat * 0.65),
    );
  }
  home() {
    if (!this.state) return;
    this.setMode("orbit");
    const h = sample(this.state.height, 0.5, 0.5) * ALTITUDE;
    this.controls.target.set(0, h * 0.25, -4);
    this.camera.position.set(53, h * 0.2 + 39, 70);
    this.controls.update();
  }
  setMode(mode) {
    if (mode === this.mode) return;
    this.mode = mode;
    this.controls.enabled = mode === "orbit";
    if (mode === "walk") {
      const dir = new THREE.Vector3();
      this.camera.getWorldDirection(dir);
      this.yaw = Math.atan2(-dir.x, -dir.z);
      this.pitch = Math.asin(dir.y);
      this.camera.position.y = Math.max(
        this.camera.position.y,
        this.ground(this.camera.position.x, this.camera.position.z) + 2.2,
      );
    } else {
      const dir = new THREE.Vector3();
      this.camera.getWorldDirection(dir);
      this.controls.target.copy(this.camera.position).addScaledVector(dir, 25);
      this.controls.update();
    }
    this.onMode?.(mode);
  }
  setBrush(kind, radius) {
    this.brush = kind;
    this.radius = radius;
    this.brushRing.visible = !!kind && !!this.brushPoint;
    if (kind) this.setMode("orbit");
    this.updateBrush();
  }
  ground(x, z) {
    return this.state
      ? sample(this.state.height, x / SIZE + 0.5, z / SIZE + 0.5) * ALTITUDE
      : 0;
  }
  pick(event) {
    if (!this.terrain) return null;
    const r = this.canvas.getBoundingClientRect();
    this.mouse.set(
      ((event.clientX - r.left) / r.width) * 2 - 1,
      (-(event.clientY - r.top) / r.height) * 2 + 1,
    );
    this.ray.setFromCamera(this.mouse, this.camera);
    return this.ray.intersectObject(this.terrain)[0]?.point || null;
  }
  updateBrush() {
    if (!this.brush || !this.brushPoint || !this.state) {
      this.brushRing.visible = false;
      return;
    }
    const pt = this.brushPoint,
      r = this.radius * SIZE;
    const p = this.brushRing.geometry.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const a = (i / (p.count - 1)) * Math.PI * 2,
        x = pt.x + Math.cos(a) * r,
        z = pt.z + Math.sin(a) * r;
      p.setXYZ(i, x, this.ground(x, z) + 0.3, z);
    }
    p.needsUpdate = true;
    this.brushRing.geometry.computeBoundingSphere();
    this.brushRing.visible = true;
  }
  bind() {
    addEventListener("resize", () => this.resize());
    this.canvas.addEventListener("pointerdown", (e) => {
      this.drag = {
        x: e.clientX,
        y: e.clientY,
        lastX: e.clientX,
        lastY: e.clientY,
      };
      if (this.brush) this.controls.enabled = false;
      this.canvas.focus({ preventScroll: true });
    });
    this.canvas.addEventListener("pointermove", (e) => {
      if (this.mode === "walk" && this.drag) {
        this.yaw -= (e.clientX - this.drag.lastX) * 0.003;
        this.pitch = THREE.MathUtils.clamp(
          this.pitch - (e.clientY - this.drag.lastY) * 0.003,
          -1.3,
          1.3,
        );
        this.drag.lastX = e.clientX;
        this.drag.lastY = e.clientY;
      }
      if (this.brush) {
        this.brushPoint = this.pick(e);
        this.updateBrush();
      }
    });
    this.canvas.addEventListener("pointerleave", () => {
      this.brushRing.visible = false;
    });
    addEventListener("pointerup", (e) => {
      if (
        this.drag &&
        this.brush &&
        Math.hypot(e.clientX - this.drag.x, e.clientY - this.drag.y) < 6 &&
        e.target === this.canvas
      ) {
        const pt = this.pick(e);
        if (pt)
          this.onPick?.({
            x: THREE.MathUtils.clamp(pt.x / SIZE + 0.5, 0, 1),
            z: THREE.MathUtils.clamp(pt.z / SIZE + 0.5, 0, 1),
          });
      }
      this.drag = null;
      this.controls.enabled = this.mode === "orbit";
    });
    addEventListener("keydown", (e) => {
      if (
        ["INPUT", "TEXTAREA", "SELECT"].includes(
          document.activeElement?.tagName,
        ) ||
        document.querySelector("dialog[open]")
      )
        return;
      this.keys.add(e.code);
      if (
        [
          "KeyW",
          "KeyA",
          "KeyS",
          "KeyD",
          "KeyQ",
          "KeyE",
          "ArrowUp",
          "ArrowDown",
          "ArrowLeft",
          "ArrowRight",
          "Space",
        ].includes(e.code)
      ) {
        e.preventDefault();
        if (!this.brush) this.setMode("walk");
      }
    });
    addEventListener("keyup", (e) => this.keys.delete(e.code));
    addEventListener("blur", () => {
      this.keys.clear();
      this.drag = null;
    });
  }
  resize() {
    const w = innerWidth,
      h = innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.composer?.setSize(w, h);
  }
  animate(now) {
    requestAnimationFrame(this.animate);
    const dt = Math.min(0.06, (now - this.last) / 1000);
    this.last = now;
    this.time += dt;
    this.sky.position.copy(this.camera.position);
    this.skyUniforms.time.value = this.time;
    if (this.waterUniforms) this.waterUniforms.time.value = this.time;
    if (this.rain.visible) {
      this.rain.position.y = -((this.time * 17) % 30);
    }
    if (this.mode === "orbit") this.controls.update();
    else {
      const speed =
        (this.keys.has("ShiftLeft") || this.keys.has("ShiftRight") ? 23 : 10) *
        dt;
      const forward =
          (this.keys.has("KeyW") || this.keys.has("ArrowUp") ? 1 : 0) -
          (this.keys.has("KeyS") || this.keys.has("ArrowDown") ? 1 : 0),
        right =
          (this.keys.has("KeyD") || this.keys.has("ArrowRight") ? 1 : 0) -
          (this.keys.has("KeyA") || this.keys.has("ArrowLeft") ? 1 : 0);
      this.camera.position.x +=
        (-Math.sin(this.yaw) * forward + Math.cos(this.yaw) * right) * speed;
      this.camera.position.z +=
        (-Math.cos(this.yaw) * forward - Math.sin(this.yaw) * right) * speed;
      this.camera.position.x = THREE.MathUtils.clamp(
        this.camera.position.x,
        -SIZE * 0.48,
        SIZE * 0.48,
      );
      this.camera.position.z = THREE.MathUtils.clamp(
        this.camera.position.z,
        -SIZE * 0.48,
        SIZE * 0.48,
      );
      if (this.keys.has("KeyE")) this.camera.position.y += speed;
      if (this.keys.has("KeyQ")) this.camera.position.y -= speed;
      this.camera.position.y = Math.max(
        this.camera.position.y,
        this.ground(this.camera.position.x, this.camera.position.z) + 1.8,
      );
      this.camera.rotation.order = "YXZ";
      this.camera.rotation.set(this.pitch, this.yaw, 0);
    }
    this.composer.render();
    this.frames++;
    if (now - this.fpsStart > 650) {
      this.onFps?.(Math.round((this.frames * 1000) / (now - this.fpsStart)));
      const direction = new THREE.Vector3();
      this.camera.getWorldDirection(direction);
      this.onDirection?.(Math.atan2(direction.x, direction.z));
      this.frames = 0;
      this.fpsStart = now;
    }
  }
  async screenshot() {
    this.brushRing.visible = false;
    this.composer.render();
    return new Promise((resolve) => this.canvas.toBlob(resolve, "image/png"));
  }
}
