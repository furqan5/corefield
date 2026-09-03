# Decision verdict

**(a) Decision date.** 03 Sep 2026. The mandatory pre-E5 version was committed before access at `98dd22c`; this version includes the one completed E5 run. Labels are **(a)** verified fact, **(b)** engineering estimate with assumptions, and **(c)** inference/judgment.

## Direct answer

**(c) No ML method has validly demonstrated a win over classical NLS anywhere in this harness.** Retained E3 results show plain NN and PINN losing at every tested load and grey-box equalling NLS outside the hull. E3 later failed its split/timing/measurement-loss audit, so the defensible answer is **“no demonstrated win,” not “proof that no win is possible.”**

**(a) The only lower-RMSE cell was not an ML win.** Generic IEC gave `0.31 K` versus NLS `0.51 K` at `1.00 pu`, but was unsafe-low on all ten seeds and failed the safety condition; it lost to NLS at the other three loads.

## Method decisions

| Item | Decision | Evidence and boundary |
|---|---|---|
| Plain NN | **Reject — interim** | **(a)** Superseded E3 mean RMSE was `44.56–105.53 K`; every load/seed peak error was unsafe-low. **(c)** Stop this frozen architecture, but do not generalize from invalid confirmatory evidence. |
| PINN | **Reject — interim** | **(a)** Superseded E3 mean RMSE rose from `11.50 K` at `1.00 pu` to `88.15 K` at `1.60 pu`; every peak error was unsafe-low. Its fitted `τ_o=77.81±6.34 min` and `Δθ_hr=41.68±2.38 K` versus NLS `167.39±1.07 min` and `21.94±0.27 K`. **(c)** Reject this frozen candidate, not physics-informed learning generally. |
| Hull-gated grey-box | **Reject outside-hull decoration; retain honest fallback** | **(a)** All `40/40` embedded checks were bit-exact with NLS and flagged extrapolation. Standalone E4 in-range utility remains untested. |
| E2 scarce-reference study | **Pending — deferred, not skipped** | **(c)** E3 made further ranking low-value; completed E5 did not change that. Revisit after valid corrected E3 evidence or if the scarce-reference niche becomes material. |
| E5 conformal bounds | **Reject under the registered rule** | **(a)** In-range coverage was `97.2%` at `0.3741 K` width, but the registered 95%-in-Clopper–Pearson-CI Boolean failed; §11 requires reject. Ordinary strict-shift coverage was `100/100/0/0%` at `1.00/1.15/1.30/1.60 pu`. KDE-weighted overlap achieved `96.1%` empirical coverage; strict weighting returned unbounded limits and `0%` finite availability. **(c)** Investigate only through a newly preregistered multi-calibration-seed in-range replication; reject every finite above-hull product. |

## Safety and change conditions

**(a) No evaluated finite estimator or bound was safe throughout the retained above-nameplate cases.** NLS/grey-box were unsafe on all seeds at `1.30` and `1.60 pu`, reaching mean signed peak error `−8.48 K` at `1.60 pu`; every generic-IEC and neural peak error was unsafe-low. Ordinary E5 bounds missed all episodes in the two highest centred bands; support-aware E5 returned no finite limit outside calibration support. **(c)** That fail-closed abstention is transparent but not operationally usable.

**(c) None of these outputs should support an operational loading decision.** E1 is a reproduction, not external validation; E3 is synthetic and non-confirmatory; E5 has one calibration realization and a literal registered failure. A corrected write-once E3, standalone E4, deferred E2, or newly preregistered external/multi-seed evidence could change the bounded decision. Until then: **nowhere demonstrated.**
