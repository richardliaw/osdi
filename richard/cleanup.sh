 #!/bin/bash

 ps -aux | grep tf | awk -F  " " '/1/ {print $2}' | xargs kill -9
