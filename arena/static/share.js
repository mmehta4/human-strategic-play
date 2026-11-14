const btn = document.getElementById("copy"), field = document.getElementById("share-url");
if (btn && field) {
  btn.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(field.value); } catch (e) { field.select(); document.execCommand("copy"); }
    btn.textContent = btn.dataset.done;
  });
}
