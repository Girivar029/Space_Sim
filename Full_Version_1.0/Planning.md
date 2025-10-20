###Planning###

#Structuring#

#Physics Modules
-Gravity.py: Gravity of planets, stars, supernovas and blackholes
-Orbits.py: Orbital Velocity and Ordering or Reasoning for orbiting(for planets to revolve around bigger ones or stars or black holes.)
-Distances.py: Reasoning for distances and calculations of impact of gravity
-Other_Physics: For any remaining physics formulae and reasoning
-Collisions.py: Accurate collisions using saved data

#Camera And Input Management
-Camera.py: Takes care of camera stuff like zoom, focus, positioning, movement etc.
-Input_Management.py: Takes care of user input and assigning uses to it.

#Celestial Bodies
-Stars.py: Accurate physical measures of stars scaled down before calculations to make it easier and a more optimised system. It also states requirements for star formation and mass etc.
-Planets.py: Accurate physical measures of planets which is also scaled down, also states all its details for better collisions.
-BlackHoles_&_Supernovas.py: Similar to Stars.py

#UI & HUD
-UI.py: Shows the UI on screen
-Data.py: Saves additional data for UI or HUD

#Scenarios
-Solar.py: Simulates the solar system as a whole
-BlackHole.py: Simulates many planets with 1 blackhole
-Supernova.py: Similar to BlackHole.py
-MultiBlackHole.py: Similar to single blackhole but has multiple.
-MultiSupernova.py: Similar to MultiBlackHole.py
-Random.py: A random no of planets and stars.
-Random_BH&SN.py: Random.py but with chance of blackholes and supernovas
-Custom: User decides.

#Main
-Main.py: Stiches everything together and runs the sim.

##Working Order and Things in code##

#Gravity#

#Theories used
-Newton's law of Universal Gravitation
-Einstein's Special Relativity
-Stellar Structure Theory
-Planetary Science
-Astornomical Unit Definition
-Plummer Softening
-Schwarzschild Solution
-Post-Newtonian Approximation
-Roche Limit Theory
-Kerr Metric
-Schwarzschild Metric
-Effective potential Theory
-Tidal Force theory
-Shell Theorem
-Gravitational Wave Theory
-Peters-Mathews Formula
-Supernova Dynamics
-Accretion Disk Theory
-Eddington Limit
-Neutron Star Equation of State
-Chandrashekar Limit
-Force superposition Principle\
-Runge-Kutta Methods
-Leapfrog integration
-Adaptive timestep control
-Energy Conservation
-Angular Momentum Conservation
-Virial Theorem
-Barnes-Hut Algorithm

#Orbits#

#Theories Used
-Keplers Laws of Planetary Motion
-Orbital velocity computations
-Orbital elements conversions
-Elipse Geometry
-Anamoly Calculations
-Newtons law of universal Gravitation
-Celestial Mechanics
-Two-Body Problem
-Eccentric Anomaly and Mean Anomaly
-Runga-Kutta Numerical Integration Methods
-Leapfrog Integration
-Adaptive TimeStep Control
-Conversion Of Energy In Orbital Motion
-Virial Theorem in Orbital Systems
-Gravity Softening
-Orbital Resonance
-Hohmann Transfer Orbit Theory
-Two-dimensional Orbital Vectors
-Newtonian Mechanics Approximation
-Vectors Calculus for Orbital Elements and State Vectors
-Solution of Kepler's Equation
-Transformation between Orbital Elements and State Vectors
-Perturbation Theory
-Time Evolution of Orbital parameters through propogation
-Numerical Stability and Convergence Criteria

#Distances#

#Theories Used
-Euclidean Geometry
-Distance Metrics in 2D Plane
-Nemerical Sampling of Orbits for Minimum Approach Calculations
-Vector Algebra for Position and Velocity Operations
-Approximations of Closest Approach using Linear Extrapolation
-Collision Detection by Distance Threshold
-Clustering Theory for Proximity Groups
-Histogram and Distributor Analysis for Spatial Density Profilling
-Spatial Filtering Using Annulus and Circular Sector Geometries
-Moving Average and Smoothing Techniques for Time Series Data
-Time-stepped Trajectory Propagation for Dynamic Distance Tracking
-Radial Density Functions (RDF) Concepts for Spatial Distribution
-Spatial Overlap and Exclusion Principles
-Sorting and Ranking Algorithms for Proximity Queries
-Graph Theory Methods for Clump Detection
-Numerical Stability in Distance Calculation
-Linear Algebra for Rotation and Coordinate Transformation
-Statistical Measure Computation
-Event driven Proximity detection