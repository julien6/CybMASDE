import {
    AfterViewInit,
    Component,
    ElementRef,
    EventEmitter,
    Input,
    OnChanges,
    OnInit,
    Output,
    SimpleChanges,
    ViewChild,
} from '@angular/core';

@Component({
    selector: 'app-mpld3',
    templateUrl: './mpld3.component.html',
    styleUrls: ['./mpld3.component.css']
})
export class Mpld3Component implements OnInit, AfterViewInit, OnChanges {
    // @ViewChild('fig', { static: true }) figElement: ElementRef<HTMLCanvasElement>;

    @Input() data!: any;

    constructor() {
    }

    ngOnInit(): void {
        if (this.data !== undefined && this.data !== null) {
            this.loadMpld3("fig", this.data);
        }
    }

    ngAfterViewInit(): void {
    }

    ngOnChanges(changes: SimpleChanges): void {
    }

    // Method to dynamically load JavaScript
    loadMpld3(figDivId: string, data: any) {
        const node = document.createElement('script');
        node.textContent = `
        function mpld3_load_lib(url, callback) {
            var s = document.createElement('script');
            s.src = url;
            s.async = true;
            s.onreadystatechange = s.onload = callback;
            s.onerror = function () { console.warn("failed to load library " + url); };
            document.getElementsByTagName("head")[0].appendChild(s);
        }

        waitForElementToExist('#` + figDivId + `').then(element => {

            if (typeof (mpld3) !== "undefined" && mpld3._mpld3IsLoaded) {
                // already loaded: just create the figure
                !function (mpld3) {
                    mpld3.draw_figure("` + figDivId + `", ` + JSON.stringify(data) + `);
                }(mpld3);
            } else if (typeof define === "function" && define.amd) {
                // require.js is available: use it to load d3/mpld3
                require.config({ paths: { d3: "https://d3js.org/d3.v5" } });
                require(["d3"], function (d3) {
                    window.d3 = d3;
                    mpld3_load_lib("https://mpld3.github.io/js/mpld3.v0.5.9.js", function () {    
                        mpld3.draw_figure("` + figDivId + `", ` + JSON.stringify(data) + `);
                    });
                });
            } else {
                // require.js not available: dynamically load d3 & mpld3
                mpld3_load_lib("https://d3js.org/d3.v5.js", function () {
                    mpld3_load_lib("https://mpld3.github.io/js/mpld3.v0.5.9.js", function () {
                        mpld3.draw_figure("` + figDivId + `", ` + JSON.stringify(data) + `);      
                    });
                });
            }

        });
        `;
        document.getElementsByTagName('head')[0].appendChild(node);
    }

}
