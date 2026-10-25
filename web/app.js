/* PRAHARI responder dashboard.
 *
 * Plain ES2020, no build step, no framework, no package manager. That is a
 * deliberate demo-risk decision: a toolchain that must `npm install` before it
 * renders is one more thing that can fail on the morning of a pitch, and it
 * buys nothing for a single-screen dashboard.
 *
 * The map draws WITHOUT a basemap by default. The road network is the
 * basemap — which means the dashboard works with no internet at all, which is
 * the condition we should assume at a venue.
 */
'use strict';

const API = '';

const state = {
  district: 'barpeta',
  mode: 'foot',
  source: 'replay',
  frame: 0,
  basemap: 'off',
  frames: [],
  payload: null,
  cards: {},
  playing: false,
  playTimer: null,
  selected: null,
  lang: 'en',
  mapReady: false,
  // Sentinel-1 acquisition to show in `real` mode. The July 2024 Assam flood
  // is the event the whole project is validated against.
  date: '2024-07-16',
};

const $ = (id) => document.getElementById(id);
const fmt = (n) => (n === null || n === undefined ? '—' : Number(n).toLocaleString('en-IN'));

let map;

/* ------------------------------------------------------------------ util */

function toast(message) {
  const el = $('toast');
  el.textContent = message;
  el.hidden = false;
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.hidden = true; }, 9000);
}

async function getJSON(path, params) {
  const url = new URL(API + path, window.location.origin);
  Object.entries(params || {}).forEach(([k, v]) => {
    if (v !== null && v !== undefined) url.searchParams.set(k, v);
  });
  const res = await fetch(url);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* non-JSON error */ }
    throw new Error(`${path} → ${res.status}: ${detail}`);
  }
  return res.json();
}

function overlayURL() {
  const u = new URL(API + '/api/overlay.png', window.location.origin);
  u.searchParams.set('district', state.district);
  u.searchParams.set('source', state.source);
  u.searchParams.set('frame', state.frame);
  if (state.source === 'real') u.searchParams.set('date', state.date);
  // Cache-bust per state, or MapLibre reuses the previous frame's image.
  u.searchParams.set('_', `${state.source}${state.frame}${state.date}`);
  return u.toString();
}

/* ------------------------------------------------------------------- map */

const EMPTY_FC = { type: 'FeatureCollection', features: [] };

function baseStyle() {
  return {
    version: 8,
    // A blank glyph host would break text layers; we draw no text on the map,
    // so none is needed and none is requested. That keeps the map fully
    // offline-capable.
    sources: {},
    layers: [{ id: 'bg', type: 'background', paint: { 'background-color': '#080a10' } }],
  };
}

/* The dashboard must NOT be gated on the map.
 *
 * Originally every render ran inside map.on('load'). That means any WebGL
 * failure — a blocklisted GPU, a locked-down projector laptop, remote desktop,
 * a backgrounded tab that never composites — leaves the ENTIRE screen blank,
 * including the dispatch list and the numbers, none of which need a map at
 * all. On demo day that is a total loss instead of a degraded one.
 *
 * So: data renders immediately, and map layers are pushed separately once (or
 * if) the map becomes ready. */
function initMap(centre) {
  try {
    map = new maplibregl.Map({
      container: 'map',
      style: baseStyle(),
      center: centre,
      zoom: 9.4,
      attributionControl: false,
    });
  } catch (err) {
    console.error('MapLibre failed to initialise', err);
    showMapFallback('This browser could not start WebGL, so the map is unavailable. '
      + 'Every number, the dispatch list and the action cards still work.');
    return;
  }

  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');
  map.addControl(new maplibregl.ScaleControl({ maxWidth: 130, unit: 'metric' }), 'bottom-right');
  map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-right');

  map.on('load', () => {
    try {
      addSources();
      addLayers();
      wireInteractions();
      state.mapReady = true;
      pushLayersToMap();
    } catch (err) {
      console.error('map layer setup failed', err);
      showMapFallback('The map layers could not be built, but the data panels are live.');
    }
  });

  map.on('error', (e) => {
    // Basemap tile 404s (offline, or OSM unreachable) must not kill anything.
    if (e && e.error && /tile|fetch|network/i.test(e.error.message || '')) return;
    console.warn('map error', e && e.error);
  });

  // If the map has not come up within 12 seconds it is not going to. Say so
  // once, plainly, and leave the rest of the dashboard working.
  setTimeout(() => {
    if (!state.mapReady) {
      showMapFallback('The map did not finish loading (WebGL may be unavailable here). '
        + 'All figures, the dispatch list and the action cards below are live.');
    }
  }, 12000);
}

function showMapFallback(message) {
  if (document.getElementById('map-fallback')) return;
  const el = document.createElement('div');
  el.id = 'map-fallback';
  el.className = 'map-fallback';
  el.innerHTML = `<div class="map-fallback-inner">
      <div class="map-fallback-title">Map unavailable</div>
      <div class="map-fallback-body">${message}</div>
    </div>`;
  document.querySelector('.map-wrap').appendChild(el);
}

function pushLayersToMap() {
  if (!state.mapReady || !state.payload) return;
  const payload = state.payload;
  Object.entries(payload.layers || {}).forEach(([id, fc]) => {
    const src = map.getSource(id);
    if (src) src.setData(fc);
  });
  const depth = map.getSource('depth');
  if (depth) depth.updateImage({ url: overlayURL(), coordinates: payload.overlay.coordinates });
  if (!pushLayersToMap._fitted && payload.overlay.bbox) {
    map.fitBounds(payload.overlay.bbox, { padding: 48, duration: 0 });
    pushLayersToMap._fitted = true;
  }
}

function addSources() {
  map.addSource('depth', {
    type: 'image',
    url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
    coordinates: [[0, 0], [0, 0], [0, 0], [0, 0]],
  });
  ['roads', 'routes', 'shelters', 'settlements', 'priorities'].forEach((id) => {
    map.addSource(id, { type: 'geojson', data: EMPTY_FC });
  });
}

function addLayers() {
  map.addLayer({
    id: 'depth-layer', type: 'raster', source: 'depth',
    paint: { 'raster-opacity': 0.82, 'raster-fade-duration': 220 },
  });

  // Roads are drawn in severity order so the blocked segments sit ON TOP of
  // the clear ones. Otherwise a long dry road drawn later hides the short
  // flooded segment that is the whole point of the map.
  map.addLayer({
    id: 'roads-clear', type: 'line', source: 'roads',
    filter: ['==', ['get', 'passability'], 'clear'],
    paint: {
      'line-color': '#46536e',
      'line-width': ['interpolate', ['linear'], ['zoom'], 8, 0.5, 12, 1.6, 15, 3],
      'line-opacity': 0.85,
    },
  });
  map.addLayer({
    id: 'roads-caution', type: 'line', source: 'roads',
    filter: ['==', ['get', 'passability'], 'caution'],
    paint: {
      'line-color': '#e8b13a',
      'line-width': ['interpolate', ['linear'], ['zoom'], 8, 1.1, 12, 2.4, 15, 4],
    },
  });
  map.addLayer({
    id: 'roads-blocked', type: 'line', source: 'roads',
    filter: ['==', ['get', 'passability'], 'vehicle_blocked'],
    paint: {
      'line-color': '#f0743c',
      'line-width': ['interpolate', ['linear'], ['zoom'], 8, 1.4, 12, 3, 15, 5],
    },
  });
  map.addLayer({
    id: 'roads-fatal', type: 'line', source: 'roads',
    filter: ['==', ['get', 'passability'], 'life_threatening'],
    paint: {
      'line-color': '#e5484d',
      'line-width': ['interpolate', ['linear'], ['zoom'], 8, 1.8, 12, 3.6, 15, 6],
    },
  });

  map.addLayer({
    id: 'routes-glow', type: 'line', source: 'routes',
    paint: { 'line-color': '#7ee0ff', 'line-width': 7, 'line-opacity': 0.16, 'line-blur': 3 },
  });
  map.addLayer({
    id: 'routes-line', type: 'line', source: 'routes',
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: {
      'line-color': '#7ee0ff',
      'line-width': ['interpolate', ['linear'], ['zoom'], 8, 1.1, 12, 2.2, 15, 3.4],
      'line-opacity': 0.9,
      'line-dasharray': [2.2, 1.3],
    },
  });

  map.addLayer({
    id: 'settlements-layer', type: 'circle', source: 'settlements',
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['zoom'], 8,
        ['interpolate', ['linear'], ['get', 'population'], 0, 1.6, 40000, 5],
        13, ['interpolate', ['linear'], ['get', 'population'], 0, 4, 40000, 15]],
      'circle-color': '#8ea0c4',
      'circle-opacity': 0.5,
      'circle-stroke-width': 1,
      'circle-stroke-color': '#0b0e14',
    },
  });

  map.addLayer({
    id: 'shelters-layer', type: 'circle', source: 'shelters',
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['zoom'], 8, 3.4, 13, 9],
      // Colour encodes how full it is, so an operator reads capacity pressure
      // off the map without opening a single popup.
      'circle-color': [
        'case',
        ['!', ['get', 'viable']], '#4b5364',
        ['>=', ['get', 'utilisation'], 0.999], '#e5484d',
        ['>=', ['get', 'utilisation'], 0.85], '#e8b13a',
        '#3fbf7f',
      ],
      'circle-stroke-width': ['case', ['!', ['get', 'viable']], 2, 1.4],
      'circle-stroke-color': ['case', ['!', ['get', 'viable']], '#e5484d', '#08131f'],
      'circle-opacity': 0.95,
    },
  });

  map.addLayer({
    id: 'priorities-halo', type: 'circle', source: 'priorities',
    filter: ['==', ['get', 'is_boat_dispatch'], true],
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['zoom'], 8, 11, 13, 26],
      'circle-color': '#ff4d5e', 'circle-opacity': 0.16, 'circle-blur': 0.55,
    },
  });
  map.addLayer({
    id: 'priorities-layer', type: 'circle', source: 'priorities',
    filter: ['==', ['get', 'is_boat_dispatch'], true],
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['zoom'], 8, 4.5, 13, 11],
      'circle-color': '#ff4d5e',
      'circle-stroke-width': 2, 'circle-stroke-color': '#1a0d12',
    },
  });
}

function setBasemap(kind) {
  const hasSource = !!map.getSource('osm');
  if (kind === 'osm' && !hasSource) {
    map.addSource('osm', {
      type: 'raster',
      tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
      tileSize: 256,
      maxzoom: 19,
      attribution: '© OpenStreetMap contributors',
    });
    map.addLayer(
      { id: 'osm-layer', type: 'raster', source: 'osm',
        paint: { 'raster-opacity': 0.42, 'raster-saturation': -0.75, 'raster-brightness-max': 0.65 } },
      'depth-layer'
    );
  } else if (kind === 'off' && hasSource) {
    if (map.getLayer('osm-layer')) map.removeLayer('osm-layer');
    map.removeSource('osm');
  }
}

/* --------------------------------------------------------------- popups */

function popupHTML(title, rows, flag, flagBad) {
  const body = rows
    .filter((r) => r[1] !== null && r[1] !== undefined && r[1] !== '')
    .map((r) => `<div class="popup-row"><span>${r[0]}</span><span>${r[1]}</span></div>`)
    .join('');
  const flagHTML = flag ? `<div class="popup-flag${flagBad ? ' bad' : ''}">${flag}</div>` : '';
  return `<div class="popup-title">${title}</div>${body}${flagHTML}`;
}

function wireInteractions() {
  const clickable = ['roads-blocked', 'roads-fatal', 'roads-caution', 'roads-clear',
                     'shelters-layer', 'priorities-layer', 'routes-line', 'settlements-layer'];
  clickable.forEach((id) => {
    map.on('mouseenter', id, () => { map.getCanvas().style.cursor = 'pointer'; });
    map.on('mouseleave', id, () => { map.getCanvas().style.cursor = ''; });
  });

  ['roads-blocked', 'roads-fatal', 'roads-caution', 'roads-clear'].forEach((id) => {
    map.on('click', id, (e) => {
      const p = e.features[0].properties;
      const label = {
        clear: 'Passable', caution: 'Caution — slow, unsafe for two-wheelers',
        vehicle_blocked: 'IMPASSABLE to vehicles', life_threatening: 'LIFE-THREATENING',
      }[p.passability] || p.passability;
      new maplibregl.Popup({ closeButton: true })
        .setLngLat(e.lngLat)
        .setHTML(popupHTML(p.name || `unnamed ${p.highway || 'road'}`, [
          ['Status', label],
          ['Water depth', `${Number(p.depth_m).toFixed(2)} m`],
          ['Segment length', `${fmt(p.length_m)} m`],
          ['Class', p.highway || '—'],
        ], p.passability !== 'clear'
             ? 'Depth is the 90th percentile along the segment — a road is only as passable as its worst point.'
             : null,
           p.passability === 'life_threatening'))
        .addTo(map);
    });
  });

  map.on('click', 'shelters-layer', (e) => {
    const p = e.features[0].properties;
    const util = Math.round((p.utilisation || 0) * 100);
    new maplibregl.Popup({ closeButton: true })
      .setLngLat(e.lngLat)
      .setHTML(popupHTML(p.name, [
        ['Status', p.viable ? 'Viable' : 'UNDER WATER — withdrawn'],
        ['Assigned', `${fmt(p.load)} / ${fmt(p.capacity)} (${util}%)`],
        ['Nominal capacity', fmt(p.nominal_capacity)],
        ['Water at site', `${Number(p.flood_depth_m).toFixed(2)} m`],
        ['Type', p.kind],
      ], p.capacity_is_estimate
           ? `Capacity is ESTIMATED from building footprint at 3.5 m²/person (Sphere minimum), not surveyed. Source: ${p.capacity_source}.`
           : null,
         !p.viable))
      .addTo(map);
  });

  map.on('click', 'priorities-layer', (e) => {
    const p = e.features[0].properties;
    new maplibregl.Popup({ closeButton: true })
      .setLngLat(e.lngLat)
      .setHTML(popupHTML(`#${p.rank} ${p.settlement_name}`, [
        ['Reason', String(p.primary_reason).replace(/_/g, ' ')],
        ['People at risk', fmt(p.people_at_risk)],
        ['Settlement population', fmt(p.population)],
        ['Water depth', `${Number(p.depth_m).toFixed(2)} m`],
      ], p.action, true))
      .addTo(map);
    openCard(p.settlement_id, p.settlement_name);
  });

  map.on('click', 'routes-line', (e) => {
    const p = e.features[0].properties;
    const avoided = typeof p.avoided_roads === 'string' ? JSON.parse(p.avoided_roads || '[]') : (p.avoided_roads || []);
    new maplibregl.Popup({ closeButton: true })
      .setLngLat(e.lngLat)
      .setHTML(popupHTML(`${p.settlement_name} → ${p.shelter_name}`, [
        ['People', fmt(p.people)],
        ['Distance', `${Number(p.distance_km).toFixed(1)} km`],
        ['Time', `${Math.round(p.travel_minutes)} min`],
        ['Worst depth on route', `${Number(p.max_depth_on_route_m).toFixed(2)} m`],
        ['Detour vs shortest', `${Number(p.detour_factor).toFixed(2)}×`],
      ], p.naive_route_would_flood
           ? `An ordinary navigation app would route this through ${avoided.join(', ')}, under up to ${Number(p.naive_route_max_depth_m).toFixed(2)} m of water. We route around it.`
           : null))
      .addTo(map);
    openCard(p.settlement_id, p.settlement_name);
  });

  map.on('click', 'settlements-layer', (e) => {
    const p = e.features[0].properties;
    new maplibregl.Popup({ closeButton: true })
      .setLngLat(e.lngLat)
      .setHTML(popupHTML(p.name, [
        ['Population', fmt(p.population)],
        ['Needing evacuation', fmt(p.people_at_risk)],
        ['Water at settlement', `${Number(p.flood_depth_m).toFixed(2)} m`],
      ]))
      .addTo(map);
  });
}

/* --------------------------------------------------------------- render */

function renderDisclosure(d, payload) {
  const box = $('disclosure');
  box.className = 'disclosure ' +
    (d.is_synthetic ? 'disclosure-synthetic' : d.is_replay ? 'disclosure-replay' : 'disclosure-live');
  $('disclosure-title').textContent =
    d.is_synthetic ? 'SYNTHETIC DEMONSTRATION'
      : d.is_replay ? 'REPLAY — HISTORICAL EVENT'
        : 'REAL SATELLITE OBSERVATION';

  const notes = d.notes.slice();
  const obs = payload && payload.scenario && payload.scenario.observation;
  if (obs) {
    notes.unshift(
      `Sentinel-1 acquired ${String(obs.acquired).slice(0, 16).replace('T', ' ')} · `
      + `${Math.round(obs.coverage * 100)}% district coverage · `
      + `${obs.flood_km2} km² flood (${obs.permanent_km2} km² permanent river subtracted)`
    );
  }
  $('disclosure-body').textContent = notes.join('  ·  ');
}

function metric(label, value, sub, cls) {
  return `<div class="metric ${cls || ''}">
    <div class="metric-label">${label}</div>
    <div class="metric-value">${value}</div>
    ${sub ? `<div class="metric-sub">${sub}</div>` : ''}
  </div>`;
}

function renderHeadline(plan) {
  const h = plan.headline;
  const cov = plan.assignment.coverage;
  $('headline').innerHTML = [
    metric('Need to evacuate', fmt(h.needing_evacuation),
      `of ${fmt(plan.exposure.total_population)} in district`, 'accent'),
    metric('Cut off — no road', fmt(h.people_cut_off),
      `${h.settlements_cut_off} settlement${h.settlements_cut_off === 1 ? '' : 's'}`,
      h.people_cut_off > 0 ? 'critical' : 'good'),
    metric('Sheltered', fmt(h.people_sheltered),
      `${Math.round(cov * 100)}% of those at risk`, cov >= 0.9 ? 'good' : ''),
    metric('No capacity', fmt(h.people_unsheltered),
      'shortfall, not a routing failure', h.people_unsheltered > 0 ? 'warn' : 'good'),
    metric('Roads cut', `${h.roads_cut_km} km`,
      `${Math.round(h.roads_cut_fraction * 100)}% of the network`),
    metric('Shelters lost', fmt(h.shelters_flooded_out),
      `${h.shelters_viable} still viable`,
      h.shelters_flooded_out > 0 ? 'warn' : 'good'),
    metric('Routes steered away from flooded roads', fmt(h.routes_steered_from_flood),
      'journeys where an ordinary navigation app would have driven into water', 'wide accent'),
  ].join('');
}

function renderDispatch(plan) {
  const list = plan.boat_dispatch || [];
  $('dispatch-count').textContent = list.length;
  $('dispatch-list').innerHTML = list.length
    ? list.map((e) => `
        <div class="item critical" data-settlement="${e.settlement_id}" data-name="${e.settlement_name}"
             data-lon="${e.lon}" data-lat="${e.lat}">
          <div class="item-top">
            <span class="item-name">#${e.rank} ${e.settlement_name}</span>
            <span class="item-count">${fmt(e.people_at_risk)}</span>
          </div>
          <div class="item-meta">${fmt(e.people_at_risk)} of ${fmt(e.population)} residents in the flood footprint${
            e.depth_m > 0.01 ? ` · ${Number(e.depth_m).toFixed(2)} m at the village centre` : ''}</div>
          <div class="item-action">${e.action}</div>
        </div>`).join('')
    : `<div class="empty">No settlement is cut off at this water level.<br/>Every village still has a road route to a viable shelter.</div>`;
}

function renderShortfall(plan) {
  const list = plan.capacity_shortfall || [];
  $('shortfall-count').textContent = list.length;
  $('shortfall-list').innerHTML = list.length
    ? list.slice(0, 12).map((e) => `
        <div class="item warn" data-settlement="${e.settlement_id}" data-name="${e.settlement_name}"
             data-lon="${e.lon}" data-lat="${e.lat}">
          <div class="item-top">
            <span class="item-name">${e.settlement_name}</span>
            <span class="item-count">${fmt(e.people_at_risk)}</span>
          </div>
          <div class="item-meta">nearest shelter ${e.nearest_shelter_minutes ?? '—'} min · ${Number(e.depth_m).toFixed(2)} m</div>
        </div>`).join('')
    : `<div class="empty">Every person needing evacuation has a shelter place.</div>`;
}

function renderShelters(payload) {
  const feats = (payload.layers && payload.layers.shelters ? payload.layers.shelters.features : [])
    .map((f) => f.properties)
    .filter((p) => p.load > 0 || !p.viable)
    .sort((a, b) => (b.utilisation || 0) - (a.utilisation || 0))
    .slice(0, 14);

  $('shelter-load').innerHTML = feats.length
    ? feats.map((p) => {
        const pct = Math.min(Math.round((p.utilisation || 0) * 100), 100);
        const cls = !p.viable ? 'full' : pct >= 100 ? 'full' : pct >= 85 ? 'high' : '';
        return `<div class="shelter">
          <div class="shelter-top">
            <span class="shelter-name">${p.viable ? '' : '⚠ '}${p.name}</span>
            <span class="shelter-num">${fmt(p.load)}/${fmt(p.capacity)}</span>
          </div>
          <div class="bar ${cls}"><span style="width:${p.viable ? pct : 100}%"></span></div>
          ${p.capacity_is_estimate ? '<div class="shelter-est">capacity estimated from footprint, not surveyed</div>' : ''}
        </div>`;
      }).join('')
    : `<div class="empty">No shelter is in use at this water level.</div>`;
}

function renderProvenance(payload) {
  $('provenance').innerHTML = payload.provenance.map((p) => `
    <div class="prov-row">
      <span class="prov-key">${p.layer}</span>
      <span>${p.detail || p.source}
        ${p.is_synthetic ? '<span class="prov-tag syn">SYNTHETIC</span>' : ''}
        ${p.is_estimate && !p.is_synthetic ? '<span class="prov-tag est">ESTIMATE</span>' : ''}
      </span>
    </div>`).join('');
}

function renderStageCurve() {
  const el = $('stage-curve');
  if (!state.frames.length) { el.innerHTML = ''; return; }
  const stages = state.frames.map((f) => f.stage_m);
  const max = Math.max(...stages, 1);
  const w = 1000, h = 34;
  const pts = stages.map((s, i) => {
    const x = (i / Math.max(stages.length - 1, 1)) * w;
    const y = h - (s / max) * (h - 4) - 2;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const cx = (state.frame / Math.max(stages.length - 1, 1)) * w;
  el.innerHTML = `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <polygon points="0,${h} ${pts.join(' ')} ${w},${h}" fill="#4da3ff" opacity="0.13"/>
    <polyline points="${pts.join(' ')}" fill="none" stroke="#4da3ff" stroke-width="1.8" opacity="0.75"/>
    <line x1="${cx}" y1="0" x2="${cx}" y2="${h}" stroke="#ff8fa0" stroke-width="1.6"/>
  </svg>`;
}

/* ---------------------------------------------------------- action card */

async function openCard(settlementId, name) {
  state.selected = settlementId;
  $('card-panel').hidden = false;
  $('card-settlement').textContent = name || settlementId;
  $('card-body').textContent = 'Loading…';
  try {
    const data = await getJSON('/api/cards', {
      district: state.district, mode: state.mode, source: state.source,
      frame: state.frame, settlement: settlementId, limit: 400,
    });
    state.cards = {};
    data.cards.forEach((c) => { state.cards[c.language] = c; });
    $('card-foot').textContent = data.translation_caveat;
    renderLangTabs();
    renderCard();
  } catch (err) {
    $('card-body').textContent =
      'No action card for this settlement in the current state.\n\n' + err.message;
    $('lang-tabs').innerHTML = '';
  }
}

function renderLangTabs() {
  const names = { en: 'English', as: 'অসমীয়া', bn: 'বাংলা', hi: 'हिन्दी' };
  const langs = Object.keys(state.cards);
  if (!langs.includes(state.lang)) state.lang = langs[0] || 'en';
  $('lang-tabs').innerHTML = langs.map((l) =>
    `<button data-lang="${l}" class="${l === state.lang ? 'active' : ''}">${names[l] || l}</button>`
  ).join('');
  $('lang-tabs').querySelectorAll('button').forEach((b) => {
    b.onclick = () => { state.lang = b.dataset.lang; renderLangTabs(); renderCard(); };
  });
}

function esc(s) {
  return String(s || '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}

function renderCard() {
  const c = state.cards[state.lang];
  if (!c) { $('card-body').textContent = '—'; return; }
  const parts = [`<span class="cs-banner">${esc(c.banner)}</span>`,
                 `<span class="cs-title">${esc(c.title)}</span>`];
  if (c.kind === 'evacuate') {
    parts.push(`<span class="cs-label">${esc(c.destination ? '' : '')}</span>`);
    parts.push(`<span class="cs-dest">${esc(c.destination)}</span>`);
    parts.push(`<span class="cs-dist">${esc(c.distance_line)}</span>`);
    if (c.route_line) parts.push(`<span class="cs-route">${esc(c.route_line)}</span>`);
  } else {
    parts.push(`<span class="cs-route">${esc(c.destination)}</span>`);
  }
  if (c.warning_line) parts.push(`<span class="cs-warn">${esc(c.warning_line)}</span>`);
  parts.push(`<span class="cs-carry">${esc(c.carry_line)}<br/>${esc(c.livestock_line)}</span>`);
  $('card-body').innerHTML = parts.join('');
}

/* -------------------------------------------------------------- refresh */

let inFlight = false;

async function refresh() {
  if (inFlight) return;
  inFlight = true;
  try {
    const payload = await getJSON('/api/state', {
      district: state.district, mode: state.mode, source: state.source, frame: state.frame,
      date: state.source === 'real' ? state.date : null,
    });
    state.payload = payload;

    renderDisclosure(payload.disclosure, payload);
    renderHeadline(payload.plan);
    renderDispatch(payload.plan);
    renderShortfall(payload.plan);
    renderShelters(payload);
    renderProvenance(payload);
    wireListClicks();
    pushLayersToMap();
    $('boot').classList.add('done');
  } catch (err) {
    console.error(err);
    toast('Could not load state — ' + err.message);
    $('boot').classList.add('done');
  } finally {
    inFlight = false;
  }
}

function wireListClicks() {
  document.querySelectorAll('.item[data-settlement]').forEach((el) => {
    el.onclick = () => {
      const lon = parseFloat(el.dataset.lon), lat = parseFloat(el.dataset.lat);
      if (!Number.isNaN(lon) && !Number.isNaN(lat)) {
        map.flyTo({ center: [lon, lat], zoom: Math.max(map.getZoom(), 12.2), duration: 700 });
      }
      openCard(el.dataset.settlement, el.dataset.name);
    };
  });
}

/* ----------------------------------------------------------------- boot */

function setFrame(i) {
  state.frame = Math.max(0, Math.min(i, Math.max(state.frames.length - 1, 0)));
  $('scrubber').value = state.frame;
  const f = state.frames[state.frame];
  if (f) $('replay-frame').textContent = f.label;
  renderStageCurve();
  refresh();
}

function togglePlay() {
  state.playing = !state.playing;
  $('play').textContent = state.playing ? '❚❚' : '▶';
  clearInterval(state.playTimer);
  if (state.playing) {
    state.playTimer = setInterval(() => {
      if (state.frame >= state.frames.length - 1) { togglePlay(); return; }
      setFrame(state.frame + 1);
    }, 1100);
  }
}

function wireControls() {
  document.querySelectorAll('#mode button').forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll('#mode button').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      state.mode = b.dataset.value;
      refresh();
    };
  });
  document.querySelectorAll('#source button').forEach((b) => {
    b.onclick = async () => {
      document.querySelectorAll('#source button').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      state.source = b.dataset.value;
      // The replay scrubber is meaningless for a single satellite acquisition.
      $('timeline').style.display = state.source === 'real' ? 'none' : '';
      if (state.source === 'real') {
        toast('Building from real observations — first load fetches DEM, OSM and '
              + 'Sentinel-1, so it can take a minute. Cached after that.');
      }
      await refresh();
    };
  });
  document.querySelectorAll('#basemap button').forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll('#basemap button').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      state.basemap = b.dataset.value;
      setBasemap(state.basemap);
    };
  });
  $('district').onchange = async (e) => {
    state.district = e.target.value;
    pushLayersToMap._fitted = false;
    await loadReplay();
    setFrame(0);
  };
  $('scrubber').oninput = (e) => setFrame(parseInt(e.target.value, 10));
  $('play').onclick = togglePlay;
  $('legend-toggle').onclick = () => $('legend').classList.toggle('hidden');
  $('card-close').onclick = () => { $('card-panel').hidden = true; };

  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.key === 'ArrowRight') setFrame(state.frame + 1);
    if (e.key === 'ArrowLeft') setFrame(state.frame - 1);
    if (e.key === ' ') { e.preventDefault(); togglePlay(); }
    if (e.key === 'Escape') $('card-panel').hidden = true;
  });
}

async function loadReplay() {
  try {
    const ev = await getJSON('/api/replay', { district: state.district });
    state.frames = ev.frames;
    $('scrubber').max = Math.max(ev.frames.length - 1, 0);
    $('replay-title').textContent = ev.name;
    $('replay-disclosure').textContent = ev.disclosure;
    renderStageCurve();
  } catch (err) {
    console.warn('no replay', err);
    state.frames = [];
  }
}

async function boot() {
  wireControls();

  try {
    const health = await getJSON('/api/health');
    const model = health.hazard_model || {};
    const chip = $('engine-chip');
    // classList.add('') throws a DOMException. Guard it — this used to abort
    // boot entirely and surface a misleading "API unreachable" toast.
    if (model.usable) chip.classList.add('ok');
    $('engine-text').textContent = model.usable
      ? `U-Net checkpoint · val IoU ${Number(model.val_iou).toFixed(3)}`
      : 'Otsu baseline (no U-Net checkpoint)';
    chip.title = model.usable
      ? 'A trained U-Net checkpoint is present and in use.'
      : 'No trained checkpoint found, so the classical Otsu baseline is serving. This is the honest fallback, not a failure.';
  } catch (err) {
    $('engine-chip').classList.add('err');
    $('engine-text').textContent = 'API unreachable';
    toast('Cannot reach the PRAHARI API. Is the server running?  ' + err.message);
    $('boot').classList.add('done');
    return;
  }

  let centre = [91.05, 26.35];
  try {
    const { districts } = await getJSON('/api/districts');
    $('district').innerHTML = districts
      .map((d) => `<option value="${d.id}">${d.name}, ${d.state}</option>`).join('');
    const current = districts.find((d) => d.id === state.district) || districts[0];
    if (current) { state.district = current.id; centre = current.centroid; }
    $('district').value = state.district;
  } catch (err) {
    console.warn('district list failed', err);
  }

  await loadReplay();
  if (state.frames.length) setFrameLabelOnly(0);

  // Data first, map second. The panels do not wait on WebGL.
  refresh();
  initMap(centre);
}

function setFrameLabelOnly(i) {
  const f = state.frames[i];
  if (f) $('replay-frame').textContent = f.label;
}

boot();
