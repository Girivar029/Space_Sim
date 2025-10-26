# All the learning I did for this projects and recollections of past work will be given here.
# This will act kinda like a rough book where I learn and plan

This will include everything from Python to math formulae to physics theorems to colors and asthetics.

Plan of action(Day 2)-
- Create a working model with planets and moons.

No direct tutorial for planets and moons on youtube.
Freestyling the basics for this version - Update- Moon calculations are too complex and not usable so skipping moons and going larger in scale.



The physics module in an astrophysics simulation serves as the core computational engine that governs the behavior of celestial bodies by applying the fundamental principles of physics,chiefly gravitation and orbital mechanics,along with managing interactions such as collisions and proximity effects.At its heart lies Newton's law of universal gravitation,which describes the mutual attraction between all masses,initiating the movement and evolution of bodies in the system.The gravitational force depends inversely on the square of the distance between any two bodies and directly on their masses, thereby influencing their accelerations.

To understand how bodies move under gravity,orbital mechanics comes into play,rooted in Kepler's laws,explaining elliptical paths around a central mass,defining parameters such as semi-major axis,eccentricity,inclination,and orbital period.These orbital elements characterize the shape and position of orbits in space,governing the dynamic interplay of celestial bodies' positions and velocities at any time,effectively propagating orbits forward in simulations.

Accurate distance and proximity computations between bodies are essential for detecting potential close encounters,collisions,or gravitational influences.By calculating Euclidean distances and utilizing spatial metrics,the system identifies bodies within influence spheres or collision radii,enabling efficient management of their interactions.Clustering and density metrics provide insight into system structure and evolution,reflecting the formation of groups like asteroid belts or satellite systems around planets.

Collision handling bridges astrophysical theory and real-world outcomes by simulating encounters ranging from gentle gravitational nudges to catastrophic impacts that fragment or merge bodies.This component tracks energy and angular momentum conservation,classifying collision types and outcomes based on impact velocity,mass ratios,and material properties.The module models debris creation,escape conditions based on escape velocity,and reaccumulation behaviors,which are critical to understanding phenomena like moon formation or accretion disks.

Throughout the simulation, energy and momentum monitoring ensures that physical laws are respected over time,identifying numerical errors or instabilities.Events such as collisions,orbital resonances,or close approaches are logged to capture the evolving history of the system for  analysis and visualization.Together,these integrated physics classes enable scalable,modular,and precies modeling of complex astrophysical systems,providing a virtual laboratory to explore the universe from planetary motion to galaxy dynamics. 

#This was learned with the help of AI and the text you see above is partially AI generated