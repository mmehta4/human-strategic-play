// Reaction time: milliseconds from page load to submit, sent with the choice.
const t0 = performance.now();
document.getElementById("f").addEventListener("submit", () => {
  document.getElementById("rt").value = Math.round(performance.now() - t0);
});
