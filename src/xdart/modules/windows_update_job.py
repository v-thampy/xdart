"""Native Windows update-process custody, embedded in the external PowerShell helper.

The updater starts suspended, joins a non-breakaway job, then executes. Closing
the job kills every retained descendant; process snapshots never prove cleanup.
"""

SOURCE = r'''
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;

public sealed class XdartUpdateJob : IDisposable {
  [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
  struct Startup {
    public int cb; public string reserved, desktop, title;
    public int x,y,xSize,ySize,xChars,yChars,fill,flags;
    public short show, reservedSize; public IntPtr reservedPtr, input, output, error;
  }
  [StructLayout(LayoutKind.Sequential)]
  struct ProcessInfo { public IntPtr process, thread; public int pid, tid; }
  [StructLayout(LayoutKind.Sequential)]
  struct BasicLimits {
    public long processTime, jobTime; public uint flags;
    public UIntPtr minWorkingSet, maxWorkingSet; public uint activeLimit;
    public UIntPtr affinity; public uint priority, scheduling;
  }
  [StructLayout(LayoutKind.Sequential)]
  struct Io { public ulong readOps, writeOps, otherOps, readBytes, writeBytes, otherBytes; }
  [StructLayout(LayoutKind.Sequential)]
  struct Limits { public BasicLimits basic; public Io io; public UIntPtr processMemory,jobMemory,peakProcess,peakJob; }
  [StructLayout(LayoutKind.Sequential)]
  struct Accounting {
    public long user,kernel,periodUser,periodKernel;
    public uint faults,total,active,terminated;
  }
  [StructLayout(LayoutKind.Sequential)]
  struct Security { public int length; public IntPtr descriptor; public int inherit; }
  [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
  static extern IntPtr CreateJobObject(IntPtr attrs, string name);
  [DllImport("kernel32.dll", SetLastError=true)]
  static extern bool SetInformationJobObject(IntPtr job,int kind,ref Limits data,uint size);
  [DllImport("kernel32.dll", SetLastError=true, EntryPoint="QueryInformationJobObject")]
  static extern bool QueryAccounting(IntPtr job,int kind,out Accounting data,uint size,IntPtr returned);
  [DllImport("kernel32.dll", SetLastError=true, EntryPoint="QueryInformationJobObject")]
  static extern bool QueryMembers(IntPtr job,int kind,IntPtr data,uint size,IntPtr returned);
  [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
  static extern bool CreateProcess(string app,StringBuilder cmd,IntPtr pa,IntPtr ta,bool inherit,
    uint flags,IntPtr env,string cwd,ref Startup startup,out ProcessInfo info);
  [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
  static extern IntPtr CreateFile(string path,uint access,uint share,ref Security attrs,uint mode,uint flags,IntPtr template);
  [DllImport("kernel32.dll", SetLastError=true)] static extern bool AssignProcessToJobObject(IntPtr job,IntPtr process);
  [DllImport("kernel32.dll", SetLastError=true)] static extern uint ResumeThread(IntPtr thread);
  [DllImport("kernel32.dll", SetLastError=true)] static extern bool TerminateJobObject(IntPtr job,uint code);
  [DllImport("kernel32.dll", SetLastError=true)] static extern bool TerminateProcess(IntPtr process,uint code);
  [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetExitCodeProcess(IntPtr process,out uint code);
  [DllImport("kernel32.dll")] static extern uint WaitForSingleObject(IntPtr handle,uint ms);
  [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
  [DllImport("kernel32.dll", SetLastError=true)] static extern IntPtr OpenProcess(uint access,bool inherit,int pid);
  [DllImport("kernel32.dll", SetLastError=true)] static extern bool IsProcessInJob(IntPtr process,IntPtr job,out bool inside);
  [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
  static extern bool QueryFullProcessImageName(IntPtr process,int flags,StringBuilder name,ref int size);

  IntPtr job, process; public int Id { get; private set; }
  static void Check(bool ok) { if (!ok) throw new Win32Exception(Marshal.GetLastWin32Error()); }
  static IntPtr File(string path,uint access,uint mode) {
    Security sa=new Security(); sa.length=Marshal.SizeOf(sa); sa.inherit=1;
    IntPtr h=CreateFile(path,access,3,ref sa,mode,0x80,IntPtr.Zero);
    if (h==new IntPtr(-1)) throw new Win32Exception(Marshal.GetLastWin32Error());
    return h;
  }
  public XdartUpdateJob(string exe,string command,string stdout,string stderr) {
    IntPtr input=IntPtr.Zero, output=IntPtr.Zero, error=IntPtr.Zero, thread=IntPtr.Zero;
    try {
      job=CreateJobObject(IntPtr.Zero,null); Check(job!=IntPtr.Zero);
      Limits limits=new Limits(); limits.basic.flags=0x2000; // KILL_ON_JOB_CLOSE; no breakaway
      Check(SetInformationJobObject(job,9,ref limits,(uint)Marshal.SizeOf(limits)));
      input=File("NUL",0x80000000,3); output=File(stdout,0x40000000,2); error=File(stderr,0x40000000,2);
      Startup startup=new Startup(); startup.cb=Marshal.SizeOf(startup); startup.flags=0x100;
      startup.input=input; startup.output=output; startup.error=error;
      ProcessInfo info;
      Check(CreateProcess(exe,new StringBuilder(command),IntPtr.Zero,IntPtr.Zero,true,0x08000004,
                          IntPtr.Zero,null,ref startup,out info)); // NO_WINDOW | SUSPENDED
      process=info.process; thread=info.thread; Id=info.pid;
      Check(AssignProcessToJobObject(job,process));
      Check(ResumeThread(thread)!=0xffffffff);
    } catch {
      if (process!=IntPtr.Zero) { TerminateProcess(process,1); WaitForSingleObject(process,5000); }
      Dispose(); throw;
    } finally {
      foreach(IntPtr h in new IntPtr[]{input,output,error,thread}) if(h!=IntPtr.Zero) CloseHandle(h);
    }
  }
  public bool HasExited { get { return WaitForSingleObject(process,0)==0; } }
  public bool WaitForExit(int ms) { return WaitForSingleObject(process,(uint)ms)==0; }
  public int ExitCode { get { uint code; Check(GetExitCodeProcess(process,out code)); return unchecked((int)code); } }
  public uint ActiveCount {
    get { Accounting a; Check(QueryAccounting(job,1,out a,(uint)Marshal.SizeOf(typeof(Accounting)),IntPtr.Zero)); return a.active; }
  }
  // The hidden console's conhost.exe joins the job and leaves with its last client; it is never stopped or counted.
  static bool IsConsoleHost(IntPtr process) {
    StringBuilder name=new StringBuilder(1024); int size=name.Capacity;
    return QueryFullProcessImageName(process,0,name,ref size)
      && System.IO.Path.GetFileName(name.ToString()).Equals("conhost.exe",StringComparison.OrdinalIgnoreCase);
  }
  // Best effort, returning how many stopped: one that exits meanwhile needs no stopping, and Stop() is the backstop.
  public int StopChildren() {
    IntPtr buffer=Marshal.AllocHGlobal(65536); int stopped=0;
    try {
      Check(QueryMembers(job,3,buffer,65536,IntPtr.Zero));
      int count=Marshal.ReadInt32(buffer,4);
      for(int i=0;i<count;i++) {
        int pid=Marshal.ReadIntPtr(buffer,8+i*IntPtr.Size).ToInt32(); if(pid==Id) continue;
        IntPtr child=OpenProcess(0x101001,false,pid); if(child==IntPtr.Zero) continue;
        try {
          bool inside;
          if(IsProcessInJob(child,job,out inside) && inside && !IsConsoleHost(child)
             && TerminateProcess(child,1)) stopped++;
        }
        finally { CloseHandle(child); }
      }
    } finally { Marshal.FreeHGlobal(buffer); }
    return stopped;
  }
  public void Stop() { Check(TerminateJobObject(job,1)); }
  public void Dispose() {
    if(job!=IntPtr.Zero) { CloseHandle(job); job=IntPtr.Zero; }
    if(process!=IntPtr.Zero) { CloseHandle(process); process=IntPtr.Zero; }
  }
}
'''
