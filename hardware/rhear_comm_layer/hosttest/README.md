# Host test for rhear_dsp.h

Kept in a SUBFOLDER on purpose. The Arduino IDE compiles every `.c` and `.cpp`
in the sketch root, so a copy of `test_dsp.c` sitting next to the `.ino` gets
linked into the firmware and collides — it defines `gAlpha`, `gGateDb` and its
own `main()`:

    multiple definition of `gAlpha' ... first defined here
    collect2: error: ld returned 1 exit status

Arduino only compiles the sketch root and `src/`, so anything in here is
ignored by the build.

Run it on the laptop, not the chip:

    cd hosttest && clang -O2 -std=c11 -o /tmp/test_dsp test_dsp.c -lm && /tmp/test_dsp
