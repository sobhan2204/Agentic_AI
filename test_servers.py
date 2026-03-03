import subprocess
import sys
import time

servers = [
    "mathserver.py",
    "weather.py", 
    "translate.py",
    "websearch.py",
    "gmail.py"
]

print("Testing each MCP server individually...\n")

for server in servers:
    print(f"Testing {server}...", end=" ", flush=True)
    
    try:
        # Start server
        proc = subprocess.Popen(
            [sys.executable, "-u", server],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait 2 seconds to see if it crashes
        time.sleep(2)
        
        if proc.poll() is None:
            # Still running - try sending initialize
            proc.stdin.write('{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}\n')
            proc.stdin.flush()
            
            # Wait for response
            time.sleep(1)
            
            if proc.poll() is None:
                print(" WORKING")
                proc.terminate()
            else:
                stderr = proc.stderr.read()
                print(f" CRASHED after initialize")
                print(f"   Error: {stderr[:200]}")
        else:
            # Crashed immediately
            stderr = proc.stderr.read()
            print(f" CRASHED immediately")
            print(f"   Error: {stderr[:200]}")
            
    except Exception as e:
        print(f"EROR: {e}")
    
    print()

print("\nTest complete!")