document.addEventListener("DOMContentLoaded", function() {
    var links = document.querySelectorAll("a");
    
    links.forEach(function(link) {
        // Check if the link has a hostname and if it differs from the current domain
        if (link.hostname && link.hostname !== window.location.hostname) {
            link.target = "_blank";
            link.rel = "noopener noreferrer";
        }
    });
});
