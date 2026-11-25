# Creates "MyClip AI" on the Desktop and in the Start menu.
# The shortcuts carry the app's AppUserModelID, so a pinned taskbar icon and the
# running window are the same icon.
param([switch]$Quiet)
$ErrorActionPreference = "Stop"
$here    = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonw = Join-Path $here ".venv\Scripts\pythonw.exe"
$script  = Join-Path $here "MyClipAI.pyw"
$icon    = Join-Path $here "myclip\assets\myclip.ico"
$appId   = "MyClipAI.Desktop"

if (-not (Test-Path $pythonw)) {
  Write-Host "MyClip isn't set up yet. Run run.bat first."
  exit 1
}

$code = @"
using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;

namespace MyClipLnk {
  [ComImport, Guid("000214F9-0000-0000-C000-000000000046"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IShellLinkW {
    void GetPath([Out, MarshalAs(UnmanagedType.LPWStr)] System.Text.StringBuilder f, int c, IntPtr d, uint fl);
    void GetIDList(out IntPtr p);
    void SetIDList(IntPtr p);
    void GetDescription([Out, MarshalAs(UnmanagedType.LPWStr)] System.Text.StringBuilder s, int c);
    void SetDescription([MarshalAs(UnmanagedType.LPWStr)] string s);
    void GetWorkingDirectory([Out, MarshalAs(UnmanagedType.LPWStr)] System.Text.StringBuilder s, int c);
    void SetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] string s);
    void GetArguments([Out, MarshalAs(UnmanagedType.LPWStr)] System.Text.StringBuilder s, int c);
    void SetArguments([MarshalAs(UnmanagedType.LPWStr)] string s);
    void GetHotkey(out short h);
    void SetHotkey(short h);
    void GetShowCmd(out int c);
    void SetShowCmd(int c);
    void GetIconLocation([Out, MarshalAs(UnmanagedType.LPWStr)] System.Text.StringBuilder s, int c, out int i);
    void SetIconLocation([MarshalAs(UnmanagedType.LPWStr)] string s, int i);
    void SetRelativePath([MarshalAs(UnmanagedType.LPWStr)] string s, uint r);
    void Resolve(IntPtr h, uint f);
    void SetPath([MarshalAs(UnmanagedType.LPWStr)] string s);
  }

  [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IPropertyStore {
    void GetCount(out uint c);
    void GetAt(uint i, out PropKey k);
    void GetValue(ref PropKey k, out PropVar v);
    void SetValue(ref PropKey k, ref PropVar v);
    void Commit();
  }

  [StructLayout(LayoutKind.Sequential, Pack = 4)]
  public struct PropKey { public Guid fmtid; public uint pid; }

  [StructLayout(LayoutKind.Explicit, Size = 24)]
  public struct PropVar {
    [FieldOffset(0)] public ushort vt;
    [FieldOffset(8)] public IntPtr p;
  }

  [ComImport, Guid("00021401-0000-0000-C000-000000000046")]
  class ShellLink {}

  public static class Lnk {
    public static void Create(string lnk, string target, string args, string dir, string icon, string desc, string appId) {
      IShellLinkW link = (IShellLinkW)new ShellLink();
      link.SetPath(target);
      link.SetArguments(args);
      link.SetWorkingDirectory(dir);
      link.SetIconLocation(icon, 0);
      link.SetDescription(desc);
      try {
        IPropertyStore store = (IPropertyStore)link;
        PropKey key = new PropKey();
        key.fmtid = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3");   // System.AppUserModel.ID
        key.pid = 5;
        PropVar v = new PropVar();
        v.vt = 31;                                                     // VT_LPWSTR
        v.p = Marshal.StringToCoTaskMemUni(appId);
        store.SetValue(ref key, ref v);
        store.Commit();
        Marshal.FreeCoTaskMem(v.p);
      } catch { }                                                      // shortcut still works without it
      ((IPersistFile)link).Save(lnk, true);
    }
  }
}
"@

$args_ = '"' + $script + '"'
$desc  = "MyClip AI - turn long videos into shorts"
$targets = @(
  (Join-Path ([Environment]::GetFolderPath("Desktop")) "MyClip AI.lnk"),
  (Join-Path ([Environment]::GetFolderPath("Programs")) "MyClip AI.lnk")
)

try {
  Add-Type -TypeDefinition $code -Language CSharp
  foreach ($t in $targets) { [MyClipLnk.Lnk]::Create($t, $pythonw, $args_, $here, $icon, $desc, $appId) }
} catch {
  # Fallback: plain shortcuts (taskbar pin may show a second icon while running)
  $sh = New-Object -ComObject WScript.Shell
  foreach ($t in $targets) {
    $l = $sh.CreateShortcut($t)
    $l.TargetPath = $pythonw; $l.Arguments = $args_; $l.WorkingDirectory = $here
    $l.IconLocation = "$icon,0"; $l.Description = $desc; $l.Save()
  }
}
if (-not $Quiet) {
  Write-Host ""
  Write-Host "Done. MyClip AI is on your Desktop and in the Start menu."
  Write-Host "To pin it: open Start, find MyClip AI, right-click it, Pin to taskbar."
}
exit 0
