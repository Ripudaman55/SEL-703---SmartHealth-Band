# Wearable Health Monitoring System (SEL703)

## Overview
This repository contains the documentation and weekly progress updates for a **wearable health monitoring project** developed as part ofDeakin unit. The project focuses on building a **lightweight, microcontroller-based wearable prototype** that captures essential physiological and activity data, performs basic on-device processing (algorithm/ML where appropriate), and supports visualisation through a web-based dashboard.

> Note: This prototype is for learning and research purposes and is **not** a medical diagnostic device.

---

## Project Focus
- **Primary goal:** Practical wearable monitoring for health/wellbeing (non-diagnostic)
- **Key measurements:** Heart rate (PPG), steps/activity (accelerometer/IMU), optional temperature
- **Analytics:** Lightweight preprocessing + simple algorithm / ML model (edge-first approach)
- **Output:** Data transmission (BLE/Wi-Fi) + backend storage + web dashboard

---

## Key Deliverables (Planned / In Progress)
- Wearable firmware for sensor reading, filtering, and feature extraction
- Lightweight edge analytics (rule-based or ML model depending on feasibility)
- Data pipeline from device → gateway/backend → database
- Web dashboard to display real-time and historical trends
- Weekly progress logs + meeting notes aligned to SEL703 reporting expectations

---

## Tech Stack (High Level)
- **Hardware:** ESP32 + wearable sensors (PPG, IMU, optional temperature)
- **Embedded:** C/C++ (Arduino/ESP-IDF style)
- **Backend:** REST API + database (implementation in progress)
- **Frontend:** Web dashboard (React/Next.js planned)
- **Documentation:** Weekly logs + design notes + meeting takeaways

---

## Repository Structure


- `docs/weekly-logs/` – Week-by-week progress, tasks, reflections
- `docs/meeting-notes/` – Mentor/supervisor meeting takeaways
- `docs/design-notes/` – Scope decisions, architecture notes, component rationale

---

## Current Status
- ✅ Project scope narrowed based on mentor guidance
- ✅ Component options identified (MCU + sensors + power)
- 🔄 Research + initial algorithm/ML approach in progress
- 🔄 Architecture + prototype implementation ongoing

---

## How This Aligns With SEL703
This repository supports professional practice by demonstrating:
- Scope management and evidence-based decision making
- Iterative development with regular progress documentation
- Reflection and reporting aligned to unit expectations
- Industry-relevant IoT + embedded + software system design

---

## License / Usage
This repository is maintained for SEL703 documentation and prototype development. Reuse is allowed for learning purposes with appropriate attribution.
