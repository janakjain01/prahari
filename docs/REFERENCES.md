# References — where every number comes from

**Read this before presenting.** A judge who works in this field will ask where 0.30 m
came from. "We picked it" is a bad answer. "Vehicle stability research puts loss of
control around 0.3 m, and here is the study" is a good one.

Every threshold in the system lives in `config/<district>.yaml`, and every one of them
is listed below with its source.

---

## 1. Road passability thresholds

Configured in `config/barpeta.yaml` under `passability:`.

| Threshold | Value | Basis |
|---|---|---|
| `clear_m` | 0.15 m | Common road-agency operational guidance treats roughly 15 cm as the limit for confident driving. Below this, standing water is a nuisance rather than a hazard. |
| `vehicle_block_m` | 0.30 m | **Shand, T.D., Cox, R.J., Blacka, M.J., Smith, G.P. (2011).** *Australian Rainfall and Runoff Revision Project 10: Appropriate Safety Criteria for Vehicles.* Engineers Australia. Experimental and analytical work on vehicle stability in floodwater; small passenger vehicles begin to lose stability from roughly 0.3 m even in near-static water, and lower still once velocity is non-trivial. |
| `life_threatening_m` | 0.60 m | **Martínez-Gomariz, E., Gómez, M., Russo, B., Djordjević, S. (2017/2018).** Experimental work on human stability in flood flows. Adult instability begins well below 0.5 m once velocity is meaningful; 0.6 m standing depth is a conservative "do not send pedestrians" line. See also **Cox, R.J., Shand, T.D., Blacka, M.J. (2010),** *ARR Revision Project 10: Appropriate Safety Criteria for People.* |
| `caution_speed_factor` | 0.4 | Engineering judgement, stated as such. Movement through shallow water is slower, not impossible. Modelling caution as a hard cut would route people the long way round for no reason; modelling it as free would under-estimate travel time. |

**We deliberately choose the conservative end of every band.** In an evacuation a false
"impassable" costs a detour; a false "passable" costs lives.

**Honest note:** these criteria are derived from Australian and European experimental
work. Indian vehicle mix (far more two-wheelers and three-wheelers) is *less* stable in
floodwater, not more, so applying these limits is conservative in the right direction —
but validating them for Indian conditions is genuine future work.

---

## 2. Hydrology

| Item | Value | Basis |
|---|---|---|
| HAND method | — | **Nobre, A.D., Cuartas, L.A., Hodnett, M., Rennó, C.D., et al. (2011).** *Height Above the Nearest Drainage — a hydrologically relevant new terrain model.* Journal of Hydrology 404(1–2), 13–29. |
| Depression filling | — | **Planchon, O. & Darboux, F. (2002)**, and **Barnes, R., Lehman, C., Mulla, D. (2014),** *Priority-Flood: An optimal depression-filling and watershed-labeling algorithm.* Computers & Geosciences 62. Implemented via morphological reconstruction, which is the same result in compiled code. |
| Flat resolution | — | **Garbrecht, J. & Martz, L.W. (1997).** *The assignment of drainage direction over flat surfaces in raster digital elevation models.* Journal of Hydrology 193. Essential here: the Brahmaputra floodplain is almost perfectly flat, and without flat resolution D8 produces parallel garbage. |
| `drainage_threshold_km2` | 50 km² | Channel-initiation threshold. Expressed as **area, not cell count**, so it means the same thing at any grid resolution. Tuned so the drainage network picks out genuine rivers and major distributaries rather than field drains. **This is the single most sensitive parameter in the system** — a much smaller value collapses HAND toward zero across the plain and makes every flood look catastrophic. |
| `max_depth_m` | 8.0 m | Plausibility clamp. Depths above this are almost always a DEM artefact or misclassified radar shadow; they are clamped and marked down in confidence rather than trusted. |

---

## 3. SAR flood detection

| Item | Value | Basis |
|---|---|---|
| Tiled Otsu | — | **Otsu, N. (1979).** *A threshold selection method from gray-level histograms.* IEEE Trans. SMC 9(1). Tile selection follows **Chini, M., Hostache, R., Giustarini, L., Matgen, P. (2017)**, *A hierarchical split-based approach for parametric thresholding of SAR images*, IEEE TGRS 55(12), and **Martinis, S., Twele, A., Voigt, S. (2009)**, *Towards operational near real-time flood detection using a split-based automatic thresholding procedure*, NHESS 9. |
| Speckle filter | Lee, 5×5 | **Lee, J.S. (1980).** *Digital image enhancement and noise filtering by use of local statistics.* IEEE PAMI 2(2). `cu = 0.523` corresponds to single-look intensity data. |
| `max_water_threshold_db` | −13.0 dB | Physical ceiling on what may be called water. Open water at C-band VV sits below roughly −15 dB. **This parameter is load-bearing:** without it, tiles of pure land are bimodal for a different reason (dark bare fields vs radar-bright built-up double-bounce), Otsu splits *those*, and the system reports more flooding at low water than at high. That failure was observed during development and is now a regression test. |
| `fallback_threshold_db` | −16.0 dB | Conventional C-band VV open-water cut, used only when no tile is bimodal. |
| Training data | — | **Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020).** *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.* CVPR Workshops. |
| Segmentation architecture | U-Net | **Ronneberger, O., Fischer, P., Brox, T. (2015).** *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI. |
| Permanent water | occurrence ≥ 80% | **Pekel, J.-F., Cottam, A., Gorelick, N., Belward, A.S. (2016).** *High-resolution mapping of global surface water and its long-term changes.* Nature 540, 418–422. Threshold set high deliberately: on a braided river the channel migrates, and a moderate threshold would mark seasonally-wet sandbar as permanent, erasing real flooding over the chars where people live. |

---

## 4. Shelter capacity

| Item | Value | Basis |
|---|---|---|
| `sqm_per_person` | 3.5 m² | **Sphere Association (2018).** *The Sphere Handbook: Humanitarian Charter and Minimum Standards in Humanitarian Response*, 4th edition — minimum covered floor area per person. Indian relief camps under surge routinely run tighter than this; we use the standard and label the figure as an estimate rather than quietly using a number that flatters our capacity. |
| `usable_floor_fraction` | 0.65 | Engineering judgement, stated as such. A school's footprint is not one open hall — corridors, walls, toilets and staff rooms are not sleeping space. Deliberately conservative: over-estimating capacity leaves people standing outside a full building in the rain. |
| `capacity_reserve_fraction` | 0.10 | Space held back for staff, stores and medical use. |
| `plinth_height_m` | 0.30 m | Public buildings in flood-prone Assam are routinely built on a raised plinth, so ground-level water does not immediately mean water inside. A local-practice adjustment, surfaced in the API response rather than hidden. |

**The honest position on capacity:** surveyed shelter capacity is not open data. Every
capacity in this system is estimated from OSM building footprint. The dashboard renders
estimates differently from surveyed figures, and production deployment requires the
district relief-camp register from ASDMA. That is a real deployment dependency, and
naming it is more credible than inventing numbers.

---

## 5. Evacuation model

| Item | Value | Basis |
|---|---|---|
| `walking_speed_kmph` | 4.0 | Standard adult walking pace. Conservative for a mixed group carrying possessions; a family with children and livestock moves slower. |
| `vehicle_speed_kmph` | 25.0 | Rural Assam roads under stress, not highway speed. |
| `max_travel_minutes` | 90 | Soft limit — journeys beyond it are penalised, not forbidden, because sometimes the only shelter *is* two hours away. |
| Min-cost flow | — | **Transportation problem**, solved exactly by network simplex. Implementation: `networkx.min_cost_flow`. See **Ahuja, R.K., Magnanti, T.L., Orlin, J.B. (1993),** *Network Flows: Theory, Algorithms, and Applications.* |

---

## 6. Alerting

| Item | Basis |
|---|---|
| CAP 1.2 | **OASIS (2010).** *Common Alerting Protocol Version 1.2.* The schema SACHET itself uses. Element order is an `xsd:sequence` and is part of the specification; polygon coordinates are `lat,lon`, the opposite of GeoJSON. |
| Alert fatigue | The documented failure mode of existing mass alerting is not delivery but people ceasing to read messages that tell them nothing actionable. This is why the citizen card carries one destination, one route, and one explicit negative instruction. |

---

## 7. Data sources

| Source | Licence / access |
|---|---|
| Copernicus DEM GLO-30 | Free, via Microsoft Planetary Computer STAC |
| Sentinel-1 RTC / GRD | Free, via Microsoft Planetary Computer STAC. **No key, no waitlist** |
| JRC Global Surface Water | Free, via Planetary Computer |
| OpenStreetMap | ODbL — attribution required |
| WorldPop 2020 UN-adjusted constrained | CC BY 4.0 |
| Open-Meteo | Free, keyless |
| Google Flood Hub | CC BY 4.0 — **API key waitlisted** |
| IMD API | **Registration + IP whitelisting** |
| Sen1Floods11 | CC BY 4.0 |

---

## A note on citation integrity

Every reference above is a real, checkable publication. Where a number is **engineering
judgement rather than a citation** — `caution_speed_factor`, `usable_floor_fraction`,
`plinth_height_m` — it is labelled as such in this table and in the code.

Do not describe a judgement call as a cited value in the pitch. If asked about one, the
correct answer is: *"That one is our assumption, here is the reasoning, and here is what
it would take to validate it."* That answer builds more credibility than a fabricated
citation destroys.
