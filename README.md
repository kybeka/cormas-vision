# cormas-vision

## Overview
This project is part of Google Summer of Code (GSoC) 2025 with the Pharo Consortium and CIRAD. It aims to integrate a real-time **computer vision system** with **Cormas**, the agent-based modeling platform developed in Pharo, to enable **physical board game interaction** with simulations.

Through this project, I am building:
- A Python-based computer vision pipeline (using OpenCV and YOLO) to detect and track physical game pieces.
- A JSON-based communication layer between the vision system and Cormas.
- Extensions to Cormas to support **live updates of board state** from physical interaction.
- A **generic gaming interface** to allow Cormas models to interact with real-world boards.

All contributions in this repository are part of an open-source effort to extend the Cormas platform and will be shared with the community.

## Key Objectives
- Develop a **board calibration tool** and real-time board tracking system.
- Implement object detection using YOLO, with Chess as a prototype.
- Build a JSON-based bridge to Cormas for real-time synchronization.
- Design modular, reusable components to generalize to other grid-based games.
- Contribute directly to the Cormas open-source ecosystem with well-documented code and tutorials.

