import subprocess
import os

def ExecCmd(cmd):
  process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  # Wait for the process to finish and get the output
  stdout, stderr = process.communicate()
  output = stdout.decode().strip()

  # Print the output and any errors
  print("Output:", output)
  if stderr:
      print("Errors:", stderr.decode())


if __name__ == "__main__":
    os.chdir("/root/my_code/neuralangelo")
    for SEQUENCE in ["dtu_scan105",  "dtu_scan110",  "dtu_scan118",  "dtu_scan55",  "dtu_scan65",  "dtu_scan83",
            "dtu_scan106",  "dtu_scan114",  "dtu_scan122", "dtu_scan40", "dtu_scan63",  "dtu_scan69",  "dtu_scan97"]:
        COLMAP_DATA_PATH=f"/root/autodl-tmp/data/dtu/{SEQUENCE}"
        SCENE_TYPE = "object"
        CMD = f"bash /root/my_code/neuralangelo/projects/neuralangelo/scripts/run_colmap_dtu.sh {COLMAP_DATA_PATH}"

        # execute COLMAP from python
        ExecCmd(CMD)