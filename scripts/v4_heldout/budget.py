"""Pure price/reservation constants; importing this never loads Modal decorators.

Rates checked at https://modal.com/pricing on 2026-10-16. Conservative CPU/memory
and 20% margin, not an account-wide billing hard cap.
"""
TIMEOUT = 900
GPU_RATE = (.000306 + 2 * .0000131 + 16 * .00000222) * 1.2
CPU_RATE = (.0000131 + 2 * .00000222) * 1.2
