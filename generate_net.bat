@echo off
if "%SUMO_HOME%"=="" (
  echo Please set SUMO_HOME environment variable.
  exit /b 1
)

set NET=simple.net.xml
set TRIPS=trips.trips.xml
set ROUTES=routes.rou.xml

"%SUMO_HOME%\bin\netgenerate" --grid --grid.number=2 --grid.length=200 -o %NET%

python "%SUMO_HOME%\tools\randomTrips.py" -n %NET% -o %TRIPS% -b 0 -e 3600 -p 2

"%SUMO_HOME%\bin\duarouter" -n %NET% -t %TRIPS% -o %ROUTES%

echo Generated: %NET% %TRIPS% %ROUTES%
