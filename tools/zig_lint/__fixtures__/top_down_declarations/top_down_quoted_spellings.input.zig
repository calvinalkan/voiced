const EquivalentQuotedSpellings = struct {
    fn @"\x68"() void {}

    pub fn entry() void {
        @"h"();
    }
};
