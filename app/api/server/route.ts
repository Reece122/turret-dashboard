import { spawn, type ChildProcess } from "child_process";
import { NextResponse } from "next/server";

// Keep the Python process handle on globalThis so it survives dev hot-reloads
// of this module (each reload would otherwise get a fresh module scope).
const g = globalThis as unknown as { pyProc?: ChildProcess | null };

function isRunning() {
  return !!g.pyProc && g.pyProc.exitCode === null && !g.pyProc.killed;
}

// Kill EVERY python_server.py (tracked or started by hand in a terminal) so we
// never end up with two processes fighting over the single camera device.
function killAll() {
  try {
    if (process.platform === "win32") {
      const cmd =
        "Get-CimInstance Win32_Process | " +
        "Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'python_server\\.py' } | " +
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }";
      spawn("powershell", ["-NoProfile", "-Command", cmd], { stdio: "ignore" });
    } else {
      spawn("pkill", ["-f", "python_server.py"], { stdio: "ignore" });
    }
  } catch {
    /* nothing to kill */
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export async function GET() {
  return NextResponse.json({ running: isRunning(), pid: g.pyProc?.pid ?? null });
}

export async function POST(req: Request) {
  const { action } = await req.json().catch(() => ({ action: "" }));

  if (action === "start") {
    if (isRunning()) {
      return NextResponse.json({ running: true, pid: g.pyProc!.pid });
    }
    // Single-instance guard: clear any stray/manual copy first, then give the
    // OS a moment to release the camera before spawning exactly one.
    killAll();
    await sleep(800);
    const proc = spawn("python", ["python_server.py"], {
      cwd: process.cwd(),
      stdio: "ignore",
    });
    g.pyProc = proc;
    proc.on("exit", () => {
      if (g.pyProc === proc) g.pyProc = null;
    });
    proc.on("error", () => {
      if (g.pyProc === proc) g.pyProc = null;
    });
    return NextResponse.json({ running: true, pid: proc.pid });
  }

  if (action === "stop") {
    // Kill by command line so it works even if the server was started
    // manually in a terminal (not just via this button).
    killAll();
    g.pyProc = null;
    return NextResponse.json({ running: false });
  }

  return NextResponse.json({ error: "unknown action" }, { status: 400 });
}
