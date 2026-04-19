// Typewriter effect for the hero headline. Cycles through data-phrases
// with a natural-feeling type/erase rhythm and a blinking caret.
//
// Phrases live on the element itself so the markup stays the source of
// truth; the script only animates. Respects prefers-reduced-motion — if
// the user opts out we just show the first phrase statically.
(function () {
  const el = document.querySelector(".typewriter .tw-text");
  const caret = document.querySelector(".typewriter .tw-caret");
  if (!el) return;

  let phrases;
  try {
    phrases = JSON.parse(el.dataset.phrases);
  } catch (_) {
    return;
  }
  if (!Array.isArray(phrases) || phrases.length === 0) return;

  const reduceMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;
  if (reduceMotion) {
    el.textContent = phrases[0];
    return;
  }

  const TYPE_MS = 45;      // ms per character typed
  const ERASE_MS = 22;     // ms per character erased
  const HOLD_FULL_MS = 1800;   // pause after fully typing a phrase
  const HOLD_EMPTY_MS = 250;   // pause after erasing

  let phraseIdx = 0;
  let charIdx = 0;
  let mode = "typing"; // "typing" | "erasing"

  function step() {
    const phrase = phrases[phraseIdx];

    if (mode === "typing") {
      charIdx++;
      el.textContent = phrase.slice(0, charIdx);
      if (charIdx === phrase.length) {
        mode = "erasing";
        setTimeout(step, HOLD_FULL_MS);
      } else {
        // slight humanising jitter on the typing cadence
        setTimeout(step, TYPE_MS + Math.random() * 40);
      }
      return;
    }

    // erasing
    charIdx--;
    el.textContent = phrase.slice(0, charIdx);
    if (charIdx === 0) {
      mode = "typing";
      phraseIdx = (phraseIdx + 1) % phrases.length;
      setTimeout(step, HOLD_EMPTY_MS);
    } else {
      setTimeout(step, ERASE_MS);
    }
  }

  // blinking caret
  if (caret) {
    setInterval(() => {
      caret.style.opacity = caret.style.opacity === "0" ? "1" : "0";
    }, 500);
  }

  // kick off after a short delay so the page settles
  setTimeout(step, 400);
})();
