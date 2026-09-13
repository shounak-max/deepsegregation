import paramiko
import sys

def run_remote(cmd):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("10.0.24.7", port=2222, username="saptarsi", password="Z^8mf23", timeout=15)
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode('utf-8', 'replace')
    err = stderr.read().decode('utf-8', 'replace')
    client.close()
    return out, err

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--file":
        with open(sys.argv[2], "r", encoding="utf-8") as f:
            cmd = f.read()
    elif len(sys.argv) > 1:
        cmd = " ".join(sys.argv[1:])
    else:
        cmd = "nvidia-smi"
    out, err = run_remote(cmd)
    print("STDOUT:\n" + out)
    if err:
        print("STDERR:\n" + err)
