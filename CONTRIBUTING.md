# Contributing to PlantLeaf

Thank you for your interest in PlantLeaf! Contributions of all kinds are welcome — bug reports, feature suggestions, algorithm improvements, documentation fixes, and pull requests.

---

## Reporting Bugs

1. Check the [existing issues](https://github.com/TommyVaninetti/PlantLeaf-Desktop-App/issues) to avoid duplicates.
2. Open a new issue with:
   - A clear title and description
   - Steps to reproduce the problem
   - Your OS and Python version
   - Any relevant error output or screenshots

---

## Suggesting Features

Open an issue with the label `enhancement`. Describe the use case and, if possible, link to relevant scientific literature or prior art.

---

## Submitting a Pull Request

1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes. Keep commits small and focused.
3. Run the application locally and verify your changes work:
   ```bash
   pip install -r requirements.txt
   python src/main.py
   ```
4. Open a Pull Request against `main`. In the description, explain *what* you changed and *why*.

---

## Code Style

- Python 3.10+ compatible code.
- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Use descriptive variable names — this codebase deals with signal processing concepts that must remain readable.
- Add comments where the logic is non-obvious (especially in detection algorithms and serial parsers).

---

## Areas Where Help Is Most Welcome

- **Click detection algorithm**: improving Stage 3 criteria, reducing false positives on noisy recordings.
- **Platform testing**: testing the app on Linux distributions and Windows versions beyond Win 11.
- **New species datasets**: if you have plant recordings, please get in touch via [contact](mailto:tommasovaninetti8@gmail.com).
- **Documentation**: improving or translating the user-facing documentation.

---

## Questions

For questions about the detection algorithm or experimental methodology, open a [Discussion](https://github.com/TommyVaninetti/PlantLeaf-Desktop-App/discussions) or reach out via the [website](https://plantleaf.it).

---

## License

By contributing, you agree that your contributions will be licensed under the **GNU Affero General Public License v3.0**, consistent with the rest of the project.
