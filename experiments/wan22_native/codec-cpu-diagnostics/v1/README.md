# CPU prefix comparison across platforms

GitHub Linux CI at commit `b6f5841adb6afe1f040436225c4f5bf1c29bd121` passed 54 of 55 new action/data checks. The tiny codec fixture failed exact equality between its standalone initial-image latent and its full-video prefix. All 55 checks passed on the Mac. The real Mac eight-window cache also passed all 11 recorded exact causal checks.

The [original Linux failure](linux-ci-failure.log) is retained. A [separate diagnostic](../../codec_cpu_diagnostic.py) compares the outputs with oneDNN enabled and disabled, at one and two CPU threads. It records maximum errors, differing element counts, input hashes and the unchanged production source. On the Mac all four configurations were exact; oneDNN is unavailable in that Torch build. [Mac measurements](mac-diagnostic.json).

The Linux diagnostic is pending. No cause is asserted yet. No codec source, actual cache, production equality check or measured source binding was changed to make this test pass.
