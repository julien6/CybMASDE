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
import { Ace, edit } from 'ace-builds';
import 'ace-builds';
import 'ace-builds/src-noconflict/theme-dracula';
@Component({
  selector: 'app-editor',
  templateUrl: './editor.component.html',
  styleUrls: ['./editor.component.css']
})
export class EditorComponent implements OnInit, AfterViewInit, OnChanges {
  @ViewChild('editor') editorRef!: ElementRef;
  @Output() textChange = new EventEmitter<string>();
  @Input() text!: string;
  @Input() readOnly: boolean = false;
  @Input() mode: string = 'json';
  @Input() terminalDisplay: boolean = false;
  @Input() prettify: boolean = true;
  editor!: Ace.Editor;
  // All possible options can be found at:
  // https://github.com/ajaxorg/ace/wiki/Configuring-Ace
  options: any = {
    highlightActiveLine: true,
    tabSize: 2,
    wrap: true,
    fontSize: 14,
    readOnly: false,
    fontFamily: '\'Roboto Mono Regular\', monospace',
  };
  constructor() { }
  ngOnInit(): void { }
  ngAfterViewInit(): void {
    this.initEditor_();
  }
  onTextChange(text: string): void {
    this.textChange.emit(text);
  }
  ngOnChanges(changes: SimpleChanges): void {
    if (!this.editor) {
      return;
    }
    for (const propName in changes) {
      if (changes.hasOwnProperty(propName)) {
        switch (propName) {
          case 'text':
            this.onExternalUpdate_();
            break;
          case 'mode':
            this.onEditorModeChange_();
            break;
          default:
        }
      }
    }
  }
  private initEditor_(): void {
    this.editor = edit(this.editorRef.nativeElement);

    this.options["showLineNumbers"] = !this.terminalDisplay;
    this.options["showGutter"] = !this.terminalDisplay;

    this.editor.setOptions(this.options);
    this.editor.setValue(this.text, -1);
    this.editor.setReadOnly(this.readOnly || this.terminalDisplay);
    this.editor.setTheme('ace/theme/dracula');
    this.setEditorMode_();
    this.editor.session.setUseWorker(false);
    this.editor.on('change', () => this.onEditorTextChange_());
  }
  private onExternalUpdate_(): void {
    const point = this.editor.getCursorPosition();
    this.editor.setValue(this.text, -1);
    this.editor.moveCursorToPosition(point);
  }
  private onEditorTextChange_(): void {
    this.text = this.editor.getValue();
    this.onTextChange(this.text);
  }
  private onEditorModeChange_(): void {
    this.setEditorMode_();
  }
  private setEditorMode_(): void {
    this.editor.getSession().setMode(`ace/mode/${this.mode}`);
  }
}