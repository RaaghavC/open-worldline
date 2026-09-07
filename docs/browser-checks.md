# Browser verification

September 7, 2026, on the local Mac in the Codex in-app browser. These are manual interface checks, separate from the Python model and storage tests.

| Workflow | Observed result |
| --- | --- |
| Generate alpine, desert and alien presets | Each produces a displayed landscape from the local trained generator |
| Apply terrain edits | Brush changes appear in the scene and create saved revisions |
| Change rain and advance the world | Learned ecology changes appear and simulation time advances |
| Create an alternate future | New world starts from the saved state and can be changed separately |
| Restore an earlier revision | Earlier state returns and the restore appears as a new revision |
| Reload the page and restart the server | Saved world and history remain available |
| Export, then import with the browser file picker | Downloaded JSON reloads with identical state and checksum |
| Save a screenshot | Browser downloads the rendered scene as a PNG |
| Switch Orbit and Explore | Instructions and active control mode change |

The export/import check initially found that browser JSON serialization changed `0.0` and `1.0` to `0` and `1`, producing a different checksum for equal state values. The final version gives equivalent numeric spellings the same checksum. Existing test worlds were migrated without changing their numeric values or revision numbers. A repeated browser export/import check compared three copies and found identical states and checksums.

The page displayed about 120 render frames per second during several observations on this computer. This is a display-counter observation, not a timed performance benchmark, and is separate from neural inference. The final CPU generation benchmark measured 208-402 ms for the three terrain categories while the browser was open. Slower hardware, more objects, background work and different cameras can change performance.

No browser console errors or warnings were observed during the checked workflows. Touch interaction, small-screen layouts, assistive-technology use and lengthy walking sessions have not been fully tested. The scene remains a stylized terrain renderer, with limited materials, objects and interaction types.
