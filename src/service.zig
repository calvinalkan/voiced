const config = @import("service/config.zig");
const control = @import("service/control.zig");
const Supervisor = @import("service/supervisor.zig");

pub const ConfigDiagnostic = config.Diagnostic;
pub const Options = Supervisor.ServiceOptions;
pub const Request = control.Request;

pub const load = config.load;
pub const run = Supervisor.runService;
pub const send = control.sendRequest;
