import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AgentsPixelsComponent } from './agents-pixels.component';

describe('AgentsPixelsComponent', () => {
  let component: AgentsPixelsComponent;
  let fixture: ComponentFixture<AgentsPixelsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ AgentsPixelsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AgentsPixelsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
