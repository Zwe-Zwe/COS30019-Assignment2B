# Introduction

## Problem Overview
Traffic congestion and incident management are critical challenges in modern urban planning. A minor accident can cause cascading delays if traffic is not effectively rerouted. The Traffic Incident Classification Problem requires a system that can not only find the shortest path between two points but also dynamically adapt to real-time incident data extracted from visual feeds (CCTV).

## System Architecture
Our solution, the ICS, integrates two major branches of Artificial Intelligence:
1.  Search & Planning (Part A): Using graph traversal algorithms (A*, Dijkstra, Yen's K-Shortest) to compute optimal paths in a weighted node-graph.
2.  Machine Learning (Part B): Using Convolutional Neural Networks (CNNs) and Transfer Learning to "see" accidents in images and quantify their impact on traffic flow.

## Scope
The system operates on the Kuching Heritage Map dataset. It processes images to detect three classes of accidents:
-   Minor (Class 01): minimal delay (1.2x time cost).
-   Moderate (Class 02): noticeable delay (1.6x time cost).
-   Severe (Class 03): major blockage (3.0x time cost).
