// Only adjacent braced arms require a blank line.
// Simple arms may touch either kind; extra separators remain allowed.

// ─── All Pairs ──────────────────────────────────────────────────────

fn validSimpleSimple(value: u8) void {
    switch (value) {
        0 => work(),
        else => work(),
    }
}

fn validSimpleBlock(value: u8) void {
    switch (value) {
        0 => work(),
        else => {
            work();
        },
    }
}

fn validBlockSimple(value: u8) void {
    switch (value) {
        0 => {
            work();
        },
        else => work(),
    }
}

fn invalidBlockBlock(value: u8) void {
    switch (value) {
        0 => {
            work();
        },

        else => {
            work();
        },
    }
}

// ─── All Triples ────────────────────────────────────────────────────

fn validSimpleSimpleSimple(value: u8) void {
    switch (value) {
        0 => work(),
        1 => work(),
        else => work(),
    }
}

fn validSimpleSimpleBlock(value: u8) void {
    switch (value) {
        0 => work(),
        1 => work(),
        else => {
            work();
        },
    }
}

fn validSimpleBlockSimple(value: u8) void {
    switch (value) {
        0 => work(),
        1 => {
            work();
        },
        else => work(),
    }
}

fn invalidSimpleBlockBlock(value: u8) void {
    switch (value) {
        0 => work(),
        1 => {
            work();
        },

        else => {
            work();
        },
    }
}

fn validBlockSimpleSimple(value: u8) void {
    switch (value) {
        0 => {
            work();
        },
        1 => work(),
        else => work(),
    }
}

fn validBlockSimpleBlock(value: u8) void {
    switch (value) {
        0 => {
            work();
        },
        1 => work(),
        else => {
            work();
        },
    }
}

fn invalidBlockBlockSimple(value: u8) void {
    switch (value) {
        0 => {
            work();
        },

        1 => {
            work();
        },
        else => work(),
    }
}

fn invalidBlockBlockBlock(value: u8) void {
    switch (value) {
        0 => {
            work();
        },

        1 => {
            work();
        },

        else => {
            work();
        },
    }
}

// ─── Separated Blocks ───────────────────────────────────────────────

fn validSeparatedBlocks(value: u8) void {
    switch (value) {
        0 => {
            work();
        },

        1 => {
            work();
        },

        else => {
            work();
        },
    }
}

fn validOptionalSeparators(value: u8) void {
    switch (value) {
        0 => work(),

        1 => {
            work();
        },

        else => work(),
    }
}

// A comment belongs to the following arm; it cannot replace the blank line.
fn invalidCommentWithoutSeparator(value: u8) void {
    switch (value) {
        0 => {
            work();
        },

        // Fallback work.
        else => {
            work();
        },
    }
}

fn validCommentAfterSeparator(value: u8) void {
    switch (value) {
        0 => {
            work();
        },

        // Fallback work.
        else => {
            work();
        },
    }
}

// ─── Block Shapes And Boundaries ────────────────────────────────────

// Empty braces are still blocks, even on one line.
fn invalidBlockEmptyBlock(value: u8) void {
    switch (value) {
        0 => {
            work();
        },

        else => {},
    }
}

fn invalidEmptyBlockBlock(value: u8) void {
    switch (value) {
        0 => {},

        else => {
            work();
        },
    }
}

fn invalidEmptyBlocks(value: u8) void {
    switch (value) {
        0 => {},

        else => {},
    }
}

fn validSimpleEmptySimple(value: u8) void {
    switch (value) {
        0 => work(),
        1 => {},
        else => work(),
    }
}

fn invalidLabeledBlocks(value: u8) u8 {
    return switch (value) {
        0, 1 => first: {
            break :first 1;
        },

        else => fallback: {
            break :fallback 2;
        },
    };
}

fn invalidInlineBlocks(value: u8) void {
    switch (value) {
        inline 0, 1 => {
            work();
        },

        inline else => {
            work();
        },
    }
}

fn validEmptySwitch(value: noreturn) void {
    switch (value) {}
}

fn validSingleSimple(value: u8) void {
    switch (value) {
        else => work(),
    }
}

fn validSingleBlock(value: u8) void {
    switch (value) {
        else => {
            work();
        },
    }
}

fn validSingleEmptyBlock(value: u8) void {
    switch (value) {
        else => {},
    }
}
