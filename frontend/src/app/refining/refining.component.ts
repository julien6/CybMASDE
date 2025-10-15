import { Component, Input, Output, EventEmitter } from '@angular/core';
import { ConfigEditorService } from '../config-editor.service';

@Component({
  selector: 'app-refining',
  templateUrl: './refining.component.html',
  styleUrls: ['./refining.component.scss']
})
export class RefiningComponent {
  @Input() refining: any;
  @Output() refiningChange = new EventEmitter<any>();

  constructor(private editorConfigService: ConfigEditorService) { }

  /**
   * Emits changes upward whenever a value is modified.
   */
  onValueChange(field: string, value: any) {
    this.editorConfigService.setValueByPath(this.refining, field, value).then((data) => {
      this.refining = data;
      this.refiningChange.emit(this.refining);
    });
  }
}
