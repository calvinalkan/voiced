// Optional blank lines are preserved, not normalized.

fn validSeparatedSimpleArms(value: u8) void {
    switch (value) {
        0 => work(),

        1 => work(),

        else => work(),
    }
}

fn validExtraBlankLines(value: u8) void {
    switch (value) {
        0 => work(),

        1 => work(),
        else => work(),
    }
}

fn validSeparatedSimpleBlockSimple(value: u8) void {
    switch (value) {
        0 => work(),

        1 => {
            work();
        },

        else => work(),
    }
}

fn validSeparatedBlockSimpleBlock(value: u8) void {
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
