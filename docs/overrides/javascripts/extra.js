document$.subscribe(function() {
    document.querySelectorAll('.doc-object span.doc-object-name').forEach(el => {
        const text = el.textContent;
        const lastDotIndex = text.lastIndexOf('.');

        if (lastDotIndex !== -1) {
            // Split path and name
            let pathStr = text.slice(0, lastDotIndex).replaceAll(".", ".&#8203;") + ".&#8203;";
            const nameStr = text.slice(lastDotIndex + 1);
            
            // Reconstruct HTML
            el.innerHTML = `<span class="doc-object-path">${pathStr}</span><span class="doc-name">${nameStr}</span>`;
        } else {
            el.innerHTML = `<span class="doc-name">${text}</span>`;
        }
    });

});