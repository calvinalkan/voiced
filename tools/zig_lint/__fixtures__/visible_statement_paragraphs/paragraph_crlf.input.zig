fn spacing() void {
	work();
 	 
	// Keep this comment and its trailing spaces.  

	defer cleanup();
	work();
}

fn work() void {}
fn cleanup() void {}
fn eof() void { work(); defer cleanup(); work(); }