# Contributing to Vyakti

## Development Workflow

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/vyakti.git
   cd vyakti
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow the code organization principles in `CLAUDE.md`
   - Add tests for new functionality
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   cargo test --workspace
   cargo clippy --all-targets
   cargo fmt --all
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**

## Code Style

- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Follow Rust API guidelines: https://rust-lang.github.io/api-guidelines/

## Commit Convention

Follow conventional commits:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding tests
- `refactor`: Code refactoring
- `perf`: Performance improvements

## Testing

- Write unit tests for all public APIs
- Add integration tests for end-to-end workflows
- Use benchmarks for performance-critical code

## Documentation

- Add doc comments to all public items
- Include examples in doc comments
- Update README.md for user-facing changes

## Questions?

Open an issue or discussion on GitHub.
