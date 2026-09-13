"""Helper: run commands on remote and return output. Used to avoid PowerShell quoting issues."""
import paramiko

HOST, PORT, USER, PASSWD = '10.0.24.7', 2222, 'saptarsi', 'Z^8mf23'

def ssh(cmd, timeout=20):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWD, timeout=timeout)
    _, out, err = c.exec_command(cmd)
    stdout = out.read().decode()
    stderr = err.read().decode()
    c.close()
    return stdout, stderr

if __name__ == '__main__':
    import sys
    import json

    action = sys.argv[1] if len(sys.argv) > 1 else 'status'

    if action == 'kill':
        o, e = ssh('pkill -9 -f train_pointnet_remote.py 2>/dev/null; echo killed')
        print(o.strip() or 'no process found')

    elif action == 'status':
        o, e = ssh('cat ~/deepsegregation/pipeline_status.json 2>/dev/null || echo {}')
        try:
            d = json.loads(o)
            print(f"Status : {d.get('status','?')}")
            print(f"Model  : {d.get('model','?')}")
            print(f"Epoch  : {d.get('current_epoch','?')} / {d.get('epochs_total','?')}")
            m = d.get('latest_metrics') or {}
            print(f"mIoU   : {m.get('mIoU',0)*100:.1f}%  (target {d.get('targets',{}).get('target_miou',0)*100:.0f}%)")
            print(f"Acc    : {m.get('accuracy',0)*100:.1f}%  (target {d.get('targets',{}).get('target_accuracy',0)*100:.0f}%)")
            print(f"ValLoss: {d.get('val_loss') or 0:.4f}  TrainLoss: {d.get('train_loss') or 0:.4f}")
            print(f"Best   : mIoU={d.get('best_val_miou',0)*100:.1f}% at ep{d.get('best_epoch')}")
            print(f"LR     : {d.get('current_lr')}")
            print(f"Elapsed: {d.get('elapsed_seconds',0)/60:.1f} min")
            per = m.get('per_class_iou', [])
            if per:
                classes = d.get('class_names', ['c0','c1','c2','c3','c4'])
                for cn, iou in zip(classes, per):
                    print(f"  {cn:20s}: IoU={iou*100:.1f}%")
        except Exception as ex:
            print('Raw:', o[:400], '| err:', ex)

    elif action == 'log':
        n = sys.argv[2] if len(sys.argv) > 2 else '40'
        o, e = ssh(f'tail -{n} ~/deepsegregation/train.log 2>/dev/null')
        print(o)

    elif action == 'deps':
        o, e = ssh(
            'source ~/miniconda3/etc/profile.d/conda.sh && conda activate deepseg_k80 && '
            'python -c "from sklearn.neighbors import KDTree; import scipy; '
            'print(\'sklearn ok\'); print(\'scipy\', scipy.__version__)"'
        )
        print(o or e)

    elif action == 'disk':
        o, e = ssh('df -h /home && du -sh ~/deepsegregation/* 2>/dev/null')
        print(o)
