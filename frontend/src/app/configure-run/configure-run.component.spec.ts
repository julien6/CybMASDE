import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ConfigureRunComponent } from './configure-run.component';

describe('ConfigureRunComponent', () => {
  let component: ConfigureRunComponent;
  let fixture: ComponentFixture<ConfigureRunComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ConfigureRunComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ConfigureRunComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
