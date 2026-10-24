#pragma once

#include "MediaSource.h"

// What the frame server activates for the virtual camera: hands out the media source.
struct Activator : winrt::implements<Activator, CBaseAttributes<IMFActivate>>
{
public:
    // IMFActivate
    STDMETHOD(ActivateObject)(REFIID riid, void** object) override;
    STDMETHOD(ShutdownObject)() override;
    STDMETHOD(DetachObject)() override;

    HRESULT Initialize();

private:
    winrt::com_ptr<MediaSource> _source;
};
