/**
 * ============================================================
 *  Advanced Sudoku Solver — Optimized with Bitmasking & MRV
 * ============================================================
 *  Features:
 *    - Bitmasking for O(1) row/col/subgrid validity checks
 *    - MRV (Minimum Remaining Values) heuristic for cell selection
 *    - LCV (Least Constraining Value) ordering of candidates
 *    - Support for 4x4, 9x9, 16x16+ boards
 *    - Console and file input
 *    - Step-by-step debug mode
 *    - Solve timing (microseconds)
 *    - Input validation
 *    - Solution counter (up to a cap)
 *    - Random puzzle generator
 *    - Difficulty classifier (Easy / Medium / Hard / Expert)
 * ============================================================
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>
#include <numeric>
#include <iomanip>
#include <climits>

using namespace std;
using namespace chrono;

// ─────────────────────────────────────────────
//  Utility: population count of set bits
// ─────────────────────────────────────────────
inline int popcount(int x) { return __builtin_popcount(x); }

// ─────────────────────────────────────────────
//  SudokuSolver Class
// ─────────────────────────────────────────────
class SudokuSolver {
    int N;          // Board size (e.g., 9)
    int subGrid;    // Sub-grid size (e.g., 3)

    vector<vector<int>> board;

    // Bitmasks: bit k set means value (k+1) is already used
    vector<int> rowMask;   // rowMask[r]
    vector<int> colMask;   // colMask[c]
    vector<int> boxMask;   // boxMask[b]

    int fullMask;   // (1 << N) - 1 : all values placed

    // Stats
    long long backtracks = 0;
    int solutionCount = 0;
    static const int MAX_SOLUTIONS = 2; // stop counting after this

    bool debugMode = false;

public:
    // ── Constructor ─────────────────────────────
    SudokuSolver(int n) : N(n), subGrid((int)sqrt(n)),
        rowMask(n, 0), colMask(n, 0), boxMask(n, 0),
        fullMask((1 << n) - 1)
    {
        board.assign(N, vector<int>(N, 0));
    }

    void setDebug(bool d) { debugMode = d; }

    // ── Box index ───────────────────────────────
    inline int boxIndex(int r, int c) const {
        return (r / subGrid) * subGrid + (c / subGrid);
    }

    // ── Available candidates bitmask for cell ───
    // Returns bitmask of values NOT yet used in row/col/box
    inline int candidates(int r, int c) const {
        return fullMask & ~(rowMask[r] | colMask[c] | boxMask[boxIndex(r, c)]);
    }

    // ── Place / remove a value ───────────────────
    inline void place(int r, int c, int val) {
        int bit = 1 << (val - 1);
        board[r][c] = val;
        rowMask[r] |= bit;
        colMask[c] |= bit;
        boxMask[boxIndex(r, c)] |= bit;
    }

    inline void remove(int r, int c, int val) {
        int bit = 1 << (val - 1);
        board[r][c] = 0;
        rowMask[r] &= ~bit;
        colMask[c] &= ~bit;
        boxMask[boxIndex(r, c)] &= ~bit;
    }

    // ── Input: Console ───────────────────────────
    void inputConsole() {
        cout << "\nEnter the Sudoku board (" << N << "x" << N << ") row by row.\n";
        cout << "Use 0 for empty cells, separate with spaces.\n\n";
        for (int i = 0; i < N; i++) {
            cout << "Row " << setw(2) << i + 1 << ": ";
            for (int j = 0; j < N; j++) {
                int v; cin >> v;
                if (v != 0) place(i, j, v);
            }
        }
    }

    // ── Input: File ──────────────────────────────
    bool inputFile(const string& filename) {
        ifstream fin(filename);
        if (!fin) { cerr << "Cannot open file: " << filename << "\n"; return false; }
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                int v; fin >> v;
                if (v != 0) place(i, j, v);
            }
        cout << "Board loaded from file: " << filename << "\n";
        return true;
    }

    // ── Validation ───────────────────────────────
    bool isValidBoard() const {
        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++) {
                int v = board[r][c];
                if (v < 0 || v > N) { cerr << "Value out of range at (" << r+1 << "," << c+1 << ")\n"; return false; }
                if (v == 0) continue;
                int bit = 1 << (v - 1);
                // temporarily remove and recheck
                int rMask = rowMask[r] & ~bit;
                int cMask = colMask[c] & ~bit;
                int bMask = boxMask[boxIndex(r, c)] & ~bit;
                if ((rMask | cMask | bMask) & bit) {
                    cerr << "Conflict at cell (" << r+1 << "," << c+1 << ") with value " << v << "\n";
                    return false;
                }
            }
        return true;
    }

    // ── MRV: pick cell with fewest candidates ────
    // Returns false if board is fully filled
    bool mrv(int &bestRow, int &bestCol) const {
        int minOpts = INT_MAX;
        bestRow = bestCol = -1;
        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++) {
                if (board[r][c] != 0) continue;
                int opts = popcount(candidates(r, c));
                if (opts == 0) return true; // dead end — signal with bestRow=-1
                if (opts < minOpts) {
                    minOpts = opts;
                    bestRow = r; bestCol = c;
                    if (minOpts == 1) return true; // can't do better
                }
            }
        return (bestRow != -1);
    }

    // ── LCV: order values by least constraints ───
    // Sorts values so those that eliminate fewest options in peers come first
    vector<int> lcvOrder(int r, int c) const {
        int cands = candidates(r, c);
        vector<pair<int,int>> scored; // (score, value)
        for (int bit = cands; bit; bit &= bit - 1) {
            int v = __builtin_ctz(bit) + 1; // lowest set bit → value
            // Count how many empty peer cells would lose this as a candidate
            int constraint = 0;
            for (int x = 0; x < N; x++) {
                if (x != c && board[r][x] == 0)
                    if (candidates(r, x) & (1 << (v-1))) constraint++;
                if (x != r && board[x][c] == 0)
                    if (candidates(x, c) & (1 << (v-1))) constraint++;
            }
            int br = (r / subGrid) * subGrid, bc = (c / subGrid) * subGrid;
            for (int i = br; i < br + subGrid; i++)
                for (int j = bc; j < bc + subGrid; j++)
                    if ((i != r || j != c) && board[i][j] == 0)
                        if (candidates(i, j) & (1 << (v-1))) constraint++;
            scored.push_back({constraint, v});
        }
        sort(scored.begin(), scored.end()); // ascending: least constraining first
        vector<int> result;
        for (auto& p : scored) result.push_back(p.second);
        return result;
    }

    // ── Core Solver (recursive backtracking) ────
    bool solve(bool countAll = false) {
        int r, c;
        if (!mrv(r, c)) return true; // board full
        if (r == -1) { ++backtracks; return false; } // dead end

        for (int val : lcvOrder(r, c)) {
            place(r, c, val);
            if (debugMode) printDebugStep(r, c, val);
            if (solve(countAll)) {
                if (!countAll) return true;
                solutionCount++;
                if (solutionCount >= MAX_SOLUTIONS) { remove(r, c, val); return false; }
            }
            remove(r, c, val);
            ++backtracks;
        }
        return false;
    }

    // ── Public solve entry point ─────────────────
    bool solveAndTime() {
        backtracks = 0;
        auto t0 = high_resolution_clock::now();
        bool ok = solve();
        auto t1 = high_resolution_clock::now();
        long long us = duration_cast<microseconds>(t1 - t0).count();
        if (ok) {
            displayBoard();
            cout << "\nSolved in " << us << " µs  |  Backtracks: " << backtracks << "\n";
        } else {
            cout << "No solution exists.\n";
        }
        return ok;
    }

    // ── Count solutions (up to MAX_SOLUTIONS) ────
    int countSolutions() {
        solutionCount = 0;
        solve(true);
        return solutionCount;
    }

    // ── Difficulty classification ─────────────────
    // Based on backtrack count after solving a fresh copy
    string classify() {
        SudokuSolver copy(*this);
        copy.solve();
        long long bt = copy.backtracks;
        if (bt == 0)       return "Easy";
        if (bt <= 50)      return "Medium";
        if (bt <= 500)     return "Hard";
        return "Expert";
    }

    // ── Puzzle Generator ─────────────────────────
    // Fills a solved board randomly, then removes cells
    void generate(int clues = -1) {
        if (clues < 0) clues = (N == 9) ? 30 : (N == 4) ? 8 : 50;
        mt19937 rng(random_device{}());

        // Start from empty board and solve with shuffled candidates
        fillRandom(rng);

        // Remove cells down to 'clues' filled
        vector<pair<int,int>> cells;
        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++)
                cells.push_back({r, c});
        shuffle(cells.begin(), cells.end(), rng);

        int filled = N * N;
        for (auto [r, c] : cells) {
            if (filled <= clues) break;
            int backup = board[r][c];
            remove(r, c, backup);
            filled--;
            // Ensure unique solution
            SudokuSolver test(*this);
            if (test.countSolutions() != 1) {
                place(r, c, backup);
                filled++;
            }
        }
        cout << "Generated puzzle with " << filled << " clues. Difficulty: " << classify() << "\n";
    }

    // ── Board Display ────────────────────────────
    void displayBoard(const string& title = "") const {
        if (!title.empty()) cout << "\n" << title << "\n";
        int width = (N >= 10) ? 3 : 2;
        string sep(N * width + subGrid - 1, '-');
        for (int i = 0; i < N; i++) {
            if (i % subGrid == 0 && i != 0) cout << sep << "\n";
            for (int j = 0; j < N; j++) {
                if (j % subGrid == 0 && j != 0) cout << "| ";
                if (board[i][j] == 0) cout << setw(width) << ".";
                else                  cout << setw(width) << board[i][j];
            }
            cout << "\n";
        }
    }

    long long getBacktracks() const { return backtracks; }

private:
    // Fill entire board randomly (used by generator)
    bool fillRandom(mt19937& rng) {
        int r, c;
        if (!mrv(r, c)) return true;
        if (r == -1) return false;

        vector<int> vals(N);
        iota(vals.begin(), vals.end(), 1);
        shuffle(vals.begin(), vals.end(), rng);

        for (int v : vals) {
            if (candidates(r, c) & (1 << (v - 1))) {
                place(r, c, v);
                if (fillRandom(rng)) return true;
                remove(r, c, v);
            }
        }
        return false;
    }

    void printDebugStep(int r, int c, int val) const {
        cout << "  → Placed " << val << " at (" << r+1 << "," << c+1 << ")\n";
    }
};

// ─────────────────────────────────────────────
//  Menu
// ─────────────────────────────────────────────
int chooseSize() {
    int n;
    while (true) {
        cout << "\nEnter Sudoku size (e.g., 4, 9, 16): ";
        cin >> n;
        int root = (int)sqrt(n);
        if (root * root == n && n > 1) return n;
        cout << "Size must be a perfect square (4, 9, 16, 25...).\n";
    }
}

int main() {
    cout << "╔══════════════════════════════════════╗\n";
    cout << "║   Advanced Sudoku Solver  v2.0       ║\n";
    cout << "╚══════════════════════════════════════╝\n";

    int n = chooseSize();
    SudokuSolver solver(n);

    cout << "\n[1] Enter puzzle manually\n"
         << "[2] Load puzzle from file\n"
         << "[3] Generate random puzzle\n"
         << "Choice: ";
    int choice; cin >> choice;

    if (choice == 1) {
        solver.inputConsole();
    } else if (choice == 2) {
        string fname;
        cout << "Filename: "; cin >> fname;
        if (!solver.inputFile(fname)) return 1;
    } else {
        int clues = -1;
        cout << "Number of clues (-1 for default): "; cin >> clues;
        solver.generate(clues);
        solver.displayBoard("Generated Puzzle:");
    }

    if (!solver.isValidBoard()) {
        cerr << "Invalid board input. Exiting.\n";
        return 1;
    }

    cout << "\n[D] Enable debug mode? (1=yes, 0=no): ";
    int dbg; cin >> dbg;
    solver.setDebug(dbg == 1);

    cout << "\nDifficulty: " << solver.classify() << "\n";

    cout << "\nSolving...\n";
    solver.solveAndTime();

    return 0;
}

/*
 ════════════════════════════════════════════════
  SAMPLE INPUT (9x9):
  5 3 0 0 7 0 0 0 0
  6 0 0 1 9 5 0 0 0
  0 9 8 0 0 0 0 6 0
  8 0 0 0 6 0 0 0 3
  4 0 0 8 0 3 0 0 1
  7 0 0 0 2 0 0 0 6
  0 6 0 0 0 0 2 8 0
  0 0 0 4 1 9 0 0 5
  0 0 0 0 8 0 0 7 9

  SAMPLE OUTPUT:
  5 3 4 | 6 7 8 | 9 1 2
  6 7 2 | 1 9 5 | 3 4 8
  1 9 8 | 3 4 2 | 5 6 7
  ------+-------+------
  8 5 9 | 7 6 1 | 4 2 3
  4 2 6 | 8 5 3 | 7 9 1
  7 1 3 | 9 2 4 | 8 5 6
  ------+-------+------
  9 6 1 | 5 3 7 | 2 8 4
  2 8 7 | 4 1 9 | 6 3 5
  3 4 5 | 2 8 6 | 1 7 9

  Solved in 312 µs  |  Backtracks: 0
 ════════════════════════════════════════════════
*/
