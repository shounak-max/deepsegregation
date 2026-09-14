import os
import paramiko
import sys

def run_remote(cmd):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    host = os.environ.get("DEEPSEGREGATION_CLUSTER_HOST", "10.0.24.7")
    port = int(os.environ.get("DEEPSEGREGATION_CLUSTER_PORT", "2222"))
    user = os.environ.get("DEEPSEGREGATION_CLUSTER_USER", "saptarsi")
    pwd = os.environ.get("DEEPSEGREGATION_CLUSTER_PASSWORD")
    client.connect(host, port=port, username=user, password=pwd, timeout=15)
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
